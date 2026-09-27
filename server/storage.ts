// ============================================================================
// STORAGE LAYER
// ----------------------------------------------------------------------------
// Abstraction over where rendered/uploaded media lives.
//
//   STORAGE_DRIVER=local (default)
//     Media is written to a local directory (Render Persistent Disk or the
//     default ./public/videos folder) and served by the Express `/videos`
//     static route. No external dependencies required.
//
//   STORAGE_DRIVER=s3
//     Media is uploaded to any S3-compatible object store
//     (AWS S3 / Cloudflare R2 / MinIO) and served from a public base URL.
//     The AWS SDK is loaded lazily so it stays an OPTIONAL dependency: if it
//     is not installed the driver transparently falls back to local mode.
//
// Environment variables (see .env.example):
//   STORAGE_DRIVER           "local" | "s3"
//   PUBLIC_DIR               directory served under /videos (local mode)
//   MEDIA_RETENTION_HOURS    auto-purge generated files older than N hours (0 = off)
//   S3_ENDPOINT              e.g. https://<accountid>.r2.cloudflarestorage.com
//   S3_REGION                e.g. "auto" (R2) or "us-east-1"
//   S3_BUCKET                bucket name
//   S3_ACCESS_KEY_ID
//   S3_SECRET_ACCESS_KEY
//   S3_PUBLIC_BASE_URL       public CDN/base URL used to build returned URLs
//   S3_FORCE_PATH_STYLE      "true" | "false"
// ============================================================================

import path from 'path';
import fs from 'fs';

export type StorageDriver = 'local' | 's3';

export interface StoredAsset {
  /** Publicly reachable URL for the asset. */
  url: string;
  /** Size of the stored asset in bytes (0 when unknown). */
  size: number;
  /** Object key used by the remote backend (S3 mode only). */
  key?: string;
}

class Storage {
  public readonly driver: StorageDriver;
  public readonly mediaDir: string;
  public readonly retentionHours: number;

  private readonly bucket?: string;
  private readonly publicBaseUrl?: string;
  private s3Client: any;

  constructor() {
    this.driver = process.env.STORAGE_DRIVER === 's3' ? 's3' : 'local';

    // Local directory that hosts the media served under /videos.
    this.mediaDir = path.resolve(process.env.PUBLIC_DIR || 'public/videos');

    const parsedRetention = Number(process.env.MEDIA_RETENTION_HOURS);
    this.retentionHours = Number.isFinite(parsedRetention) && parsedRetention > 0 ? parsedRetention : 0;

    this.bucket = process.env.S3_BUCKET;
    this.publicBaseUrl = process.env.S3_PUBLIC_BASE_URL?.replace(/\/+$/, '');

    // Make sure the local media directory always exists.
    if (!fs.existsSync(this.mediaDir)) {
      fs.mkdirSync(this.mediaDir, { recursive: true });
    }
  }

  /**
   * Build the public URL for a locally stored file.
   * Local mode -> `/videos/<filename>` (served by the Express static route).
   */
  public publicUrlFor(localPath: string): string {
    const filename = path.basename(localPath);
    if (this.driver === 's3' && this.publicBaseUrl) {
      return `${this.publicBaseUrl}/${filename}`;
    }
    return `/videos/${filename}`;
  }

  /**
   * Persist a locally rendered/uploaded file.
   * - local mode: no-op (the file already lives in the served directory).
   * - s3 mode: upload to the object store, returning its public URL + size.
   *   Falls back to local mode on any error so rendering never breaks.
   */
  public async publish(localPath: string): Promise<StoredAsset> {
    const size = this.safeSize(localPath);

    if (this.driver !== 's3') {
      return { url: this.publicUrlFor(localPath), size };
    }

    try {
      const key = path.basename(localPath);
      const client = await this.getS3Client();
      if (!client || !this.bucket) {
        throw new Error('S3 driver selected but client/bucket is unavailable');
      }

      const { PutObjectCommand } = await this.importOptional('@aws-sdk/client-s3');
      await client.send(
        new PutObjectCommand({
          Bucket: this.bucket,
          Key: key,
          Body: fs.createReadStream(localPath),
          ContentType: this.contentTypeFor(localPath),
        })
      );

      const url = this.publicBaseUrl
        ? `${this.publicBaseUrl}/${key}`
        : `/videos/${key}`;

      return { url, size, key };
    } catch (err: any) {
      console.warn(`[storage] S3 upload failed, falling back to local: ${err?.message || err}`);
      return { url: this.publicUrlFor(localPath), size };
    }
  }

  /**
   * Delete generated files (not uploads) older than retentionHours.
   * Uploaded user files are prefixed `uploaded_` and always preserved.
   */
  public cleanupOldFiles(): void {
    if (this.retentionHours <= 0) return;

    const cutoff = Date.now() - this.retentionHours * 3600 * 1000;
    let removed = 0;

    try {
      for (const entry of fs.readdirSync(this.mediaDir)) {
        if (entry.startsWith('uploaded_')) continue;
        const fullPath = path.join(this.mediaDir, entry);
        try {
          const stat = fs.statSync(fullPath);
          if (stat.isFile() && stat.mtimeMs < cutoff) {
            fs.rmSync(fullPath, { force: true });
            removed++;
          }
        } catch {
          /* ignore individual file errors */
        }
      }
      if (removed > 0) {
        console.log(`[storage] Retention cleanup removed ${removed} file(s) older than ${this.retentionHours}h`);
      }
    } catch (err: any) {
      console.warn(`[storage] Retention cleanup failed: ${err?.message || err}`);
    }
  }

  // ---- internals ----------------------------------------------------------

  private safeSize(localPath: string): number {
    try {
      return fs.statSync(localPath).size;
    } catch {
      return 0;
    }
  }

  private contentTypeFor(localPath: string): string {
    const ext = path.extname(localPath).toLowerCase();
    if (ext === '.mp4') return 'video/mp4';
    if (ext === '.jpg' || ext === '.jpeg') return 'image/jpeg';
    if (ext === '.png') return 'image/png';
    return 'application/octet-stream';
  }

  /**
   * Import an OPTIONAL dependency at runtime without letting TypeScript/esbuild
   * require it at compile time. The specifier is built dynamically so neither
   * the type-checker nor the bundler tries to resolve `@aws-sdk/client-s3`
   * unless the s3 driver is actually used.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private async importOptional(pkg: string): Promise<any> {
    const dynamicImport = new Function('specifier', 'return import(specifier)');
    return dynamicImport(pkg);
  }

  /** Lazily construct an S3-compatible client (optional dependency). */
  private async getS3Client(): Promise<any> {
    if (this.s3Client) return this.s3Client;

    try {
      const { S3Client } = await this.importOptional('@aws-sdk/client-s3');
      this.s3Client = new S3Client({
        region: process.env.S3_REGION || 'auto',
        endpoint: process.env.S3_ENDPOINT,
        forcePathStyle: process.env.S3_FORCE_PATH_STYLE === 'true',
        credentials: {
          accessKeyId: process.env.S3_ACCESS_KEY_ID || '',
          secretAccessKey: process.env.S3_SECRET_ACCESS_KEY || '',
        },
      });
      return this.s3Client;
    } catch (err: any) {
      console.warn(
        `[storage] @aws-sdk/client-s3 not installed; run "npm i @aws-sdk/client-s3" to enable S3 mode. Falling back to local.`
      );
      return null;
    }
  }
}

export const storage = new Storage();
