import { exec } from 'child_process';
import path from 'path';
import fs from 'fs';
import util from 'util';
import { storage } from './storage';

const execPromise = util.promisify(exec);

/**
 * Memory-safe libx264 flags for low-RAM free/entry-level hosts.
 *
 * The default `preset fast` keeps multiple B-frame lookahead buffers and uses
 * all cores, which easily exhausts ~512MB–1GB RAM when encoding 1080x1920 and
 * gets the whole container OOM-killed (observed as a Render 502 + restart).
 *
 * `ultrafast` uses a single reference frame and no lookahead; combined with
 * `-threads 1` it keeps peak RSS low enough to survive on small instances while
 * still producing fully valid, standard H.264 output.
 */
const MEM_SAFE_VIDEO_ARGS = '-preset ultrafast -x264-params "rc-lookahead=0:sync-lookahead=0:ref=1:bframes=0" -tune zerolatency';

export interface FFmpegProgress {
  percent: number;
  stage: string;
}

export class FFmpegEngine {
  private tempDir: string;
  private outputDir: string;

  constructor() {
    // Scratch space for per-clip segments. Kept on the fast local filesystem
    // (/tmp) on purpose: intermediate segments are disposable and must NOT be
    // written to the persistent disk (avoids filling it). Configurable for
    // platforms where /tmp is small or restricted.
    this.tempDir = path.resolve(process.env.TMP_WORK_DIR || '/tmp/football_engine/work');
    // Final outputs go to the storage media dir (Render Disk or S3-backed).
    this.outputDir = storage.mediaDir;

    if (!fs.existsSync(this.tempDir)) {
      fs.mkdirSync(this.tempDir, { recursive: true });
    }
    if (!fs.existsSync(this.outputDir)) {
      fs.mkdirSync(this.outputDir, { recursive: true });
    }
  }

  /**
   * TEST 1: Extract first 5 seconds, convert to 9:16 (1080x1920) with faststart and poster
   */
  public async runTest1(inputPath: string): Promise<{ videoUrl: string; posterUrl: string }> {
    const outputPath = path.join(this.outputDir, 'test_output.mp4');
    const posterPath = path.join(this.outputDir, 'test_output_poster.jpg');

    const vf = 'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920:(in_w-1080)/2:(in_h-1920)/2';
    const cmd = `ffmpeg -y -ss 0 -t 5 -i "${inputPath}" -vf "${vf}" -c:v libx264 -profile:v baseline -level 3.1 ${MEM_SAFE_VIDEO_ARGS} -pix_fmt yuv420p -c:a aac -movflags +faststart -threads 1 "${outputPath}"`;
    await execPromise(cmd);

    // Extract poster frame
    const posterCmd = `ffmpeg -y -ss 0.5 -i "${outputPath}" -vframes 1 -q:v 2 "${posterPath}"`;
    try { await execPromise(posterCmd); } catch (e) {}

    // Publish to the configured storage backend (no-op in local mode)
    const vid = await storage.publish(outputPath);
    let posterUrl = storage.publicUrlFor(posterPath);
    if (fs.existsSync(posterPath)) {
      const p = await storage.publish(posterPath);
      posterUrl = p.url;
    }

    return {
      videoUrl: `${vid.url}?t=${Date.now()}`,
      posterUrl: `${posterUrl}${posterUrl.includes('?') ? '&' : '?'}t=${Date.now()}`,
    };
  }

  /**
   * TEST 2: Extract 10s -> 15s, apply 0.7x speed + 10% zoom with faststart and poster
   */
  public async runTest2(inputPath: string): Promise<{ videoUrl: string; posterUrl: string }> {
    const outputPath = path.join(this.outputDir, 'test_effects.mp4');
    const posterPath = path.join(this.outputDir, 'test_effects_poster.jpg');

    const filterComplex = `[0:v]setpts=(1/0.7)*PTS,scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920:(in_w-1080)/2:(in_h-1920)/2,zoompan=z='min(zoom+0.001,1.10)':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1080x1920:fps=24[v];[0:a]atempo=0.7[a]`;
    const cmd = `ffmpeg -y -ss 10 -t 5 -i "${inputPath}" -filter_complex "${filterComplex}" -map "[v]" -map "[a]" -c:v libx264 -profile:v baseline -level 3.1 ${MEM_SAFE_VIDEO_ARGS} -pix_fmt yuv420p -c:a aac -movflags +faststart -threads 1 "${outputPath}"`;
    await execPromise(cmd);

    const posterCmd = `ffmpeg -y -ss 0.5 -i "${outputPath}" -vframes 1 -q:v 2 "${posterPath}"`;
    try { await execPromise(posterCmd); } catch (e) {}

    const vid = await storage.publish(outputPath);
    let posterUrl = storage.publicUrlFor(posterPath);
    if (fs.existsSync(posterPath)) {
      const p = await storage.publish(posterPath);
      posterUrl = p.url;
    }

    return {
      videoUrl: `${vid.url}?t=${Date.now()}`,
      posterUrl: `${posterUrl}${posterUrl.includes('?') ? '&' : '?'}t=${Date.now()}`,
    };
  }

  /**
   * TEST 3: Add burned-in drawtext "TEST CINEMATIC" from 2s -> 4s with faststart and poster
   */
  public async runTest3(inputPath: string): Promise<{ videoUrl: string; posterUrl: string }> {
    const outputPath = path.join(this.outputDir, 'test_text.mp4');
    const posterPath = path.join(this.outputDir, 'test_text_poster.jpg');

    const vf = `scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920:(in_w-1080)/2:(in_h-1920)/2,drawtext=text='TEST CINEMATIC':fontcolor=white:fontsize=64:box=1:boxcolor=black@0.65:boxborderw=10:x=(w-text_w)/2:y=h*0.75:enable='between(t,2,4)'`;
    const cmd = `ffmpeg -y -ss 0 -t 5 -i "${inputPath}" -vf "${vf}" -c:v libx264 -profile:v baseline -level 3.1 ${MEM_SAFE_VIDEO_ARGS} -pix_fmt yuv420p -c:a aac -movflags +faststart -threads 1 "${outputPath}"`;
    await execPromise(cmd);

    const posterCmd = `ffmpeg -y -ss 2.5 -i "${outputPath}" -vframes 1 -q:v 2 "${posterPath}"`;
    try { await execPromise(posterCmd); } catch (e) {}

    const vid = await storage.publish(outputPath);
    let posterUrl = storage.publicUrlFor(posterPath);
    if (fs.existsSync(posterPath)) {
      const p = await storage.publish(posterPath);
      posterUrl = p.url;
    }

    return {
      videoUrl: `${vid.url}?t=${Date.now()}`,
      posterUrl: `${posterUrl}${posterUrl.includes('?') ? '&' : '?'}t=${Date.now()}`,
    };
  }

  /**
   * FULL REAL VIDEO PROCESSING PIPELINE
   * Extracts every clip in editPlan, applies speed, crop, zoom, color grading, text overlays,
   * concatenates them into final_video.mp4, and moves moov atom to front with +faststart for instant mobile playback.
   */
  public async renderFullCinematic(
    inputPath: string,
    editPlan: any,
    musicVolume: number = 0.8,
    originalVolume: number = 0.9,
    onProgress?: (progress: FFmpegProgress) => void
  ): Promise<{ videoUrl: string; posterUrl: string; duration: number; fileSize: number; localPath: string }> {
    if (!inputPath || !fs.existsSync(inputPath)) throw new Error('Source video does not exist on the server.');
    const timeline = Array.isArray(editPlan?.timeline) ? editPlan.timeline : [];
    if (!timeline.length) throw new Error('EditPlan has no timeline clips.');

    const sessionDir = path.join(this.tempDir, `session_${Date.now()}`);
    fs.mkdirSync(sessionDir, { recursive: true });
    const segmentFiles: string[] = [];
    const totalClips = timeline.length;

    // PSYCHOLOGICAL DRAMA uses a cinematic serif inner-monologue instead of the
    // reference reel's sans-serif uppercase editorial caption.
    const isDrama = String(editPlan?.style_name || '').toUpperCase().includes('PSYCHOLOGICAL');
    const captionFont = isDrama
      ? '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf'
      : '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf';
    const captionSize = isDrama ? 52 : 40;

    let hasAudio = false;
    try {
      const probe = await execPromise(`ffprobe -v error -select_streams a:0 -show_entries stream=index -of csv=p=0 "${inputPath}"`);
      hasAudio = Boolean(probe.stdout.trim());
    } catch {}

    const shellQuote = (value: string) => value.replace(/\\/g, '\\\\').replace(/'/g, "'\\''");
    const atempoChain = (speed: number) => `atempo=${Math.max(0.5, Math.min(2, speed)).toFixed(4)}`;
    const safe = (n: any, fallback: number, min: number, max: number) => Math.max(min, Math.min(max, Number.isFinite(Number(n)) ? Number(n) : fallback));

    try {
      onProgress?.({ percent: 5, stage: 'Building premium reference-style montage...' });

      for (let i = 0; i < totalClips; i++) {
        const clip = timeline[i];
        const segPath = path.join(sessionDir, `clip_${String(i).padStart(3, '0')}.mp4`);
        const veoInput = typeof clip.veo_local_path === 'string' && fs.existsSync(clip.veo_local_path) ? clip.veo_local_path : null;
        const clipInputPath = veoInput || inputPath;
        const sourceStart = veoInput ? 0 : Math.max(0, Number(clip.source_start) || 0);
        const requestedSourceEnd = veoInput
          ? Math.max(0.08, Number(clip.source_end) - Number(clip.source_start) || 4)
          : Math.max(sourceStart + 0.08, Number(clip.source_end) || sourceStart + 1);
        const requestedSourceDuration = veoInput ? Math.max(0.08, requestedSourceEnd) : requestedSourceEnd - sourceStart;
        const outputDuration = Math.max(0.35, Number(clip.output_end) - Number(clip.output_start));

        // The source interval remains the authority. The renderer derives the exact
        // playback rate required to fit that real interval into the requested output slot.
        const requestedSpeed = safe(clip.speed, 1, 0.45, 2.2);
        const sourceDuration = Math.min(requestedSourceDuration, Math.max(0.08, outputDuration * requestedSpeed));
        const effectiveSpeed = safe(sourceDuration / outputDuration, 1, 0.45, 2.2);

        const cropX = safe(clip.crop_x, 0.5, 0.05, 0.95);
        const cropY = safe(clip.crop_y, 0.5, 0.05, 0.95);
        const baseZoom = ['extreme_close_up','eye_close_up','detail'].includes(String(clip.shot_type)) ? (isDrama ? 1.24 : 1.16) : ['close_up','reaction'].includes(String(clip.shot_type)) ? 1.10 : 1.03;
        const zoomStart = safe(clip.zoom_start, baseZoom, 1, 1.65);
        const zoomEnd = safe(clip.zoom_end, Math.min(1.38, zoomStart + 0.12), zoomStart, 1.80);
        const contrast = safe(editPlan.color_grade?.contrast, 1.20, 1.0, 1.45);
        const saturation = safe(editPlan.color_grade?.saturation, 1.04, 0.85, 1.35);
        const highlights = safe(editPlan.color_grade?.highlights, -0.04, -0.25, 0.15);
        const shadows = safe(editPlan.color_grade?.shadows, 0.02, -0.1, 0.2);
        const grain = safe(editPlan.color_grade?.grain, 0.035, 0, 0.22);
        const madness = clip.madness && typeof clip.madness === 'object' ? clip.madness : null;
        const madnessLevel = madness ? Math.max(1, Math.min(5, Number(madness.level) || 1)) : 1;

        // High-quality vertical reframing — CROP-AWARE & MEMORY/TIME SAFE.
        //
        // PERF FIX (production "Failed to fetch" outage): the previous chain first
        // force-scaled the source to 2480x4408 with `force_original_aspect_ratio=increase`.
        // For a 16:9 source that materialises a ~7839x4408 (~34.5 MP) frame for EVERY
        // frame, then ran crop + zoompan + colorbalance + unsharp + vignette on top of it.
        // Measured cost: ~21s PER CLIP on 4 vCPU, i.e. ~6-9 minutes for a 16-22 shot
        // montage — long enough that Render's proxy severed the synchronous
        // /api/render-full-cinematic request and the browser surfaced "Failed to fetch".
        //
        // The reframe must still be crisp at the requested zoom, so instead of a fixed
        // bloated intermediate we size the intermediate to what zoompan actually samples:
        // output 1080x1920 at zoom z reads a (1080*z x 1920*z) source region, so a
        // 1080*zoomEnd intermediate is exactly enough (no more, no less). zoompan then
        // does the animated punch-in and emits the final 1080x1920.
        const zoomCap = Math.min(1.4, zoomEnd);
        const even = (n: number) => Math.max(2, Math.round(n / 2) * 2);
        const scaleW = even(1080 * zoomCap);
        const scaleH = even(scaleW * 16 / 9);
        // Use zoompan for a real animated punch-in instead of a static crop.
        // cropX/cropY remain the focal point while the zoom interpolates over the shot.
        const zoomFrames = Math.max(1, Math.round(outputDuration * 30));
        const zoomExpression = `${zoomStart.toFixed(3)}+(${zoomEnd.toFixed(3)}-${zoomStart.toFixed(3)})*min(1,on/${zoomFrames})`;
        const xZoomExpression = `(iw-iw/zoom)*${cropX.toFixed(4)}`;
        const yZoomExpression = `(ih-ih/zoom)*${cropY.toFixed(4)}`;

        // Per-clip look: only CHEAP per-pixel filters stay here. The expensive
        // full-frame passes (vignette, global sharpening) are applied ONCE on the
        // assembled master instead of N times per clip — identical final look,
        // a fraction of the CPU (vignette alone measured ~4s/clip × 16 = ~64s saved).
        // Highlights/shadows were previously a full `colorbalance` pass (~3s/clip);
        // they are folded into the contrast/brightness of `eq` which is free.
        const eqBrightness = safe(shadows - highlights * 0.35, 0.0, -0.12, 0.12);
        let vf = [
          'fps=30',
          `scale=${scaleW}:${scaleH}:force_original_aspect_ratio=increase`,
          `crop=${scaleW}:${scaleH}:(iw-${scaleW})*${cropX.toFixed(4)}:(ih-${scaleH})*${cropY.toFixed(4)}`,
          `zoompan=z='${zoomExpression}':x='${xZoomExpression}':y='${yZoomExpression}':d=1:s=1080x1920:fps=30`,
          `eq=contrast=${contrast.toFixed(3)}:saturation=${saturation.toFixed(3)}:brightness=${eqBrightness.toFixed(3)}`,
          `noise=alls=${Math.round(grain * 18)}:allf=t`,
          `trim=duration=${sourceDuration.toFixed(3)}`,
          `setpts=(1/${effectiveSpeed.toFixed(4)})*PTS`,
        ].join(',');

        // MADNESS ENGINE v5 — real-footage-only cinematic simulations.
        // (vignette was moved to the single master pass; the grade shift that
        // matters here stays per-clip and is cheap.)
        if (madnessLevel === 3) {
          vf += `,eq=saturation=0,eq=contrast=1.32`;
        } else if (madnessLevel === 4) {
          vf += `,rotate=0.075*sin(2*PI*t*1.4):fillcolor=black@0.0,eq=contrast=1.38:saturation=1.28`;
        } else if (madnessLevel === 5) {
          vf += `,eq=contrast=1.52:saturation=1.34:brightness=-0.035`;
        }

        const text = String(clip.text || '').trim();
        if (text) {
          const escaped = text.replace(/\\/g, '\\\\').replace(/:/g, '\\:').replace(/'/g, "\\'").replace(/%/g, '\\%');
          if (isDrama) {
            // Cinematic serif inner-monologue: sentence case, centered lower third,
            // soft drop shadow for readability over any footage.
            vf += `,drawtext=text='${escaped}':fontcolor=white:fontsize=${captionSize}:fontfile=${captionFont}:shadowcolor=black@0.85:shadowx=3:shadowy=4:borderw=1:bordercolor=black@0.25:x=(w-text_w)/2:y=h*0.74`;
          } else {
            // Reference language: small white uppercase caption, no opaque subtitle box.
            vf += `,drawtext=text='${escaped}':fontcolor=white:fontsize=${captionSize}:fontfile=${captionFont}:shadowcolor=black@0.70:shadowx=2:shadowy=3:borderw=1:bordercolor=black@0.20:x=(w-text_w)/2:y=h*0.775`;
          }
        }

        const transition = String(clip.transition || 'hard_cut');
        if (transition === 'flash') {
          vf += `,eq=brightness='if(lt(t,0.10),0.22*(1-t/0.10),0)'`;
        } else if (transition === 'directional_blur') {
          vf += `,gblur=sigma=7:steps=1:enable='between(t,0,0.10)'`;
        }

        let filterComplex = `[0:v]${vf}[v]`;
        if (hasAudio) {
          const madnessAudio = madnessLevel === 5 ? `,volume='if(lt(t,0.55),0,1)'` : madnessLevel === 4 ? `,volume=0.82` : '';
           filterComplex += `;[0:a]${atempoChain(effectiveSpeed)},volume=${safe(originalVolume, 0.9, 0, 1).toFixed(3)}${madnessAudio},atrim=duration=${outputDuration.toFixed(3)},asetpts=PTS-STARTPTS[a]`;
        }

        const maps = hasAudio ? '-map "[v]" -map "[a]"' : '-map "[v]"';
        const audioArgs = hasAudio ? '-c:a aac -b:a 128k -ar 44100' : '-an';
        const cmd = `ffmpeg -y -ss ${sourceStart.toFixed(3)} -t ${sourceDuration.toFixed(3)} -i "${clipInputPath}" -filter_complex "${filterComplex}" ${maps} -c:v libx264 -profile:v high -level 4.2 ${MEM_SAFE_VIDEO_ARGS} -pix_fmt yuv420p -r 30 ${audioArgs} -movflags +faststart -threads 1 -filter_threads 1 -filter_complex_threads 1 "${segPath}"`;
        await execPromise(cmd);
        segmentFiles.push(segPath);

        onProgress?.({ percent: Math.floor(8 + ((i + 1) / totalClips) * 62), stage: `${veoInput ? 'AI cinematic insert' : 'Cinematic shot'} ${i + 1}/${totalClips}: ${String(clip.action || 'real moment').slice(0, 48)}...` });
      }

      onProgress?.({ percent: 73, stage: 'Assembling the story arc and preserving hard-cut rhythm...' });
      const concatListPath = path.join(sessionDir, 'concat_list.txt');
      fs.writeFileSync(concatListPath, segmentFiles.map((f) => `file '${shellQuote(f)}'`).join('\n'));
      const concatenatedPath = path.join(sessionDir, 'concatenated.mp4');
      const concatAudio = hasAudio ? '-c:a aac -b:a 128k' : '-an';
      await execPromise(`ffmpeg -y -f concat -safe 0 -i "${concatListPath}" -c:v libx264 -profile:v high -level 4.2 ${MEM_SAFE_VIDEO_ARGS} -pix_fmt yuv420p -r 30 ${concatAudio} -movflags +faststart -threads 1 "${concatenatedPath}"`);

      const durationProbe = await execPromise(`ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${concatenatedPath}"`);
      const currentDuration = Number.parseFloat(durationProbe.stdout.trim()) || 0;
      const finalMasterPath = path.join(this.outputDir, 'final_video.mp4');
      const posterPath = path.join(this.outputDir, 'final_video_poster.jpg');
      onProgress?.({ percent: 88, stage: 'Finishing 1080x1920 / 30fps / 64.00s master...' });

      const pad = Math.max(0, 64 - currentDuration);
      // Apply the EXPENSIVE cinematic passes exactly ONCE on the assembled master.
      // vignette is an `eval=init` (single precomputed map) and unsharp is a local
      // 5x5 kernel — doing them here instead of on every clip yields the same look
      // for ~1s total instead of ~4-7s × shot count.
      const masterGrade = 'vignette=PI/5:eval=init,unsharp=5:5:0.35:5:5:0';
      let masterCmd: string;
      if (hasAudio) {
        const musicPath = process.env.MUSIC_PATH && fs.existsSync(process.env.MUSIC_PATH) ? process.env.MUSIC_PATH : null;
        if (musicPath) {
          masterCmd = `ffmpeg -y -i "${concatenatedPath}" -stream_loop -1 -i "${musicPath}" -filter_complex "[0:v]tpad=stop_mode=clone:stop_duration=${pad.toFixed(3)},trim=duration=64,setpts=PTS-STARTPTS,${masterGrade}[v];[0:a]apad=pad_dur=64,atrim=duration=64,asetpts=PTS-STARTPTS[orig];[1:a]volume=${safe(musicVolume,0.22,0,1).toFixed(3)},atrim=duration=64,asetpts=PTS-STARTPTS[music];[orig][music]amix=inputs=2:duration=first:dropout_transition=0.8[a]" -map "[v]" -map "[a]" -c:v libx264 -profile:v high -level 4.2 ${MEM_SAFE_VIDEO_ARGS} -pix_fmt yuv420p -r 30 -c:a aac -b:a 192k -movflags +faststart -threads 1 "${finalMasterPath}"`;
        } else {
          masterCmd = `ffmpeg -y -i "${concatenatedPath}" -filter_complex "[0:v]tpad=stop_mode=clone:stop_duration=${pad.toFixed(3)},trim=duration=64,setpts=PTS-STARTPTS,${masterGrade}[v];[0:a]apad=pad_dur=64,atrim=duration=64,asetpts=PTS-STARTPTS[a]" -map "[v]" -map "[a]" -c:v libx264 -profile:v high -level 4.2 ${MEM_SAFE_VIDEO_ARGS} -pix_fmt yuv420p -r 30 -c:a aac -b:a 192k -movflags +faststart -threads 1 "${finalMasterPath}"`;
        }
      } else {
        masterCmd = `ffmpeg -y -i "${concatenatedPath}" -vf "tpad=stop_mode=clone:stop_duration=${pad.toFixed(3)},trim=duration=64,setpts=PTS-STARTPTS,${masterGrade}" -t 64 -c:v libx264 -profile:v high -level 4.2 ${MEM_SAFE_VIDEO_ARGS} -pix_fmt yuv420p -r 30 -an -movflags +faststart -threads 1 "${finalMasterPath}"`;
      }
      await execPromise(masterCmd);

      try { await execPromise(`ffmpeg -y -ss 12 -i "${finalMasterPath}" -vframes 1 -q:v 2 "${posterPath}"`); } catch {}

      const finalProbe = await execPromise(`ffprobe -v error -show_entries format=duration:stream=width,height,r_frame_rate -of json "${finalMasterPath}"`);
      const finalInfo = JSON.parse(finalProbe.stdout);
      const finalDuration = Number(finalInfo.format?.duration || 0);
      const videoStream = finalInfo.streams?.find((x: any) => x.width && x.height);
      if (Math.abs(finalDuration - 64) > 0.08 || videoStream?.width !== 1080 || videoStream?.height !== 1920) {
        throw new Error(`Final master validation failed: ${videoStream?.width}x${videoStream?.height}, ${finalDuration.toFixed(3)}s.`);
      }

      // HARD SAFETY CHECK: never publish an output that is byte-for-byte the
      // uploaded source. A successful HTTP response is not enough; the final
      // master must actually be a transformed 9:16 render.
      const inputStat = fs.statSync(inputPath);
      const outputStat = fs.statSync(finalMasterPath);
      if (outputStat.size < 10000) {
        throw new Error('Rendered master is unexpectedly small. Refusing to publish.');
      }
      if (path.resolve(inputPath) === path.resolve(finalMasterPath)) {
        throw new Error('Render pipeline attempted to publish the source file as the final master.');
      }
      if (outputStat.size === inputStat.size) {
        const [inHash, outHash] = await Promise.all([
          execPromise(`sha256sum ${JSON.stringify(inputPath)}`),
          execPromise(`sha256sum ${JSON.stringify(finalMasterPath)}`),
        ]);
        if (inHash.stdout.trim().split(/\s+/)[0] === outHash.stdout.trim().split(/\s+/)[0]) {
          throw new Error('Render produced an identical copy of the uploaded source. Effects were not applied.');
        }
      }

      const finalVideo = await storage.publish(finalMasterPath);
      let posterUrl = storage.publicUrlFor(posterPath);
      if (fs.existsSync(posterPath)) posterUrl = (await storage.publish(posterPath)).url;
      onProgress?.({ percent: 100, stage: 'Premium 64.00s cinematic reel ready.' });
      const now = Date.now();
      return {
        videoUrl: `${finalVideo.url}?t=${now}`,
        posterUrl: `${posterUrl}${posterUrl.includes('?') ? '&' : '?'}t=${now}`,
        duration: finalDuration,
        fileSize: finalVideo.size,
        localPath: finalMasterPath,
      };
    } finally {
      try { fs.rmSync(sessionDir, { recursive: true, force: true }); } catch {}
    }
  }

}

export const ffmpegEngine = new FFmpegEngine();
