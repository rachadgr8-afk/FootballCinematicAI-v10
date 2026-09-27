# Deployment sur Render — FOOTBALL CINEMATIC AI

Ce guide résout les **2 points critiques** bloquant la production :

1. **FFmpeg / FFprobe absents du runtime Node natif de Render** → réglé par le `Dockerfile`.
2. **Système de fichiers éphémère** (cache/tampon) → expliqué en bas, avec les options.

---

## 1. FFmpeg sur Render (solution Dockerfile)

### Pourquoi c'est nécessaire
- `server.ts` et `server/ffmpegEngine.ts` appellent directement les binaires
  `ffmpeg` et `ffprobe` via `child_process.exec`.
- Le runtime **Node natif** de Render n'embarque **pas** FFmpeg → toutes les
  routes de rendu renvoient une erreur 500 (`spawn ffmpeg ENOENT`).
- L'overlay texte (`drawtext`) a en plus besoin de **fontconfig + une police**
  (sinon : `Cannot find a valid font for the family Sans`).

### Ce que fait le Dockerfile fourni
- **Étape 1 (builder)** : installe les dépendances (`npm ci --legacy-peer-deps`),
  compile le frontend Vite (`/app/dist`) et pré-bundle le serveur
  (`esbuild server.ts -> server.js`, départs à froid rapides).
- **Étape 2 (runtime)** : `node:22-bookworm-slim` + `apt-get install ffmpeg
  fontconfig fonts-dejavu-core`. Échoue le build si `ffmpeg -version` ne
  répond pas. N'installe que les dépendances de prod, puis copie `dist`,
  `server.js` et `public/`.
- Le serveur Express sert **l'API et le SPA** sur le même port → URLs relatives
  côté frontend (`VITE_API_BASE_URL=""`), plus de problème CORS/endpoint.

> `--legacy-peer-deps` est requis : le projet épingle `vite@^8` qui entre en
> conflit de peer dependency avec la plage d'`esbuild` en résolution stricte.

---

## 2. Déploiement pas à pas sur Render

### Option A — Dashboard (recommandé, le plus rapide)
1. Render Dashboard → **New** → **Web Service** → connecte le repo Git.
2. **Language / Runtime** : `Docker`.
3. **Dockerfile Path** : `./Dockerfile`.
4. **Instance Type** : `Starter` (1 GB) minimum — le rendu FFmpeg est gourmand
   en RAM/CPU. `Standard` (2 GB) pour des rendus volumineux ou simultanés.
5. **Environment** → ajoute :
   - `NODE_ENV` = `production`
   - `GEMINI_API_KEY` = `<ta clé>` (secret, ne jamais committer)
6. **Health Check Path** : `/health`.
7. **Create Web Service**. Render build l'image et démarre.

### Option B — Blueprint (déclaratif)
Le fichier `render.yaml` est prêt : Render Dashboard → **New** → **Blueprint**
→ sélectionne le repo. Tout est déjà configuré (runtime docker, région,
healthcheck, variables). Il ne reste qu'à saisir `GEMINI_API_KEY`.

### Vérification post-déploiement
```bash
curl https://<ton-service>.onrender.com/health
# => {"success":true,"service":"fotbal-backend"}
```
Dans les logs Render, tu dois voir au démarrage :
```
[BOOT] FFmpeg detected -> ffmpeg version 7.x ...
[BOOT] FFprobe detected. Video pipeline is ready.
```

### Tester le rendu réel
```bash
curl -X POST https://<ton-service>.onrender.com/api/test-render-3 \
  -H "Content-Type: application/json" -d '{}'
# => { "success": true, "videoUrl": "/videos/test_text.mp4?t=...", ... }
```

---

## 3. Point critique n°2 — Stockage / cache éphémère ✅ RÉSOLU

Sur Render, **le filesystem du conteneur est éphémère** : tout ce qui est écrit à
l'exécution est perdu au redéploiement/redémarrage/scale. Une **couche de
stockage unifiée** (`server/storage.ts`) gère désormais les deux scénarios,
sélectionnés automatiquement au démarrage :

| Mode | `STORAGE_DRIVER` | Persistance | Idéal pour |
|---|---|---|---|
| **Disque persistant** | `local` (+ `PUBLIC_DIR`) | Survit aux redéploiements (1 instance) | Démarrer vite, valider le pipeline |
| **Stockage objet S3/R2** | `s3` (+ variables `S3_*`) | Illimitée, multi-instances | Passer à l'échelle / CDN |

### Variables d'environnement
```
STORAGE_DRIVER=local          # "local" (défaut) | "s3"
PUBLIC_DIR=/var/data/videos   # Dossier servi+écrit (monter le disque ici)
MEDIA_RETENTION_HOURS=48      # Purge auto des fichiers + vieux de N h (0 = off)

# --- Mode S3 / R2 ---
S3_ENDPOINT=https://<account>.r2.cloudflarestorage.com
S3_REGION=auto                # "auto" (R2) | "us-east-1" (AWS)
S3_BUCKET=football-media
S3_ACCESS_KEY_ID=...
S3_SECRET_ACCESS_KEY=...
S3_PUBLIC_BASE_URL=https://cdn.example.com
S3_FORCE_PATH_STYLE=false     # true pour MinIO
```

### Option A — Disque persistant Render (recommandé pour démarrer)
Déjà configuré dans `render.yaml` :
- Disque `football-media` monté sur **`/var/data`** (5 GB)
- `PUBLIC_DIR=/var/data/videos`
- `MEDIA_RETENTION_HOURS=48`

⚠️ Un disque **ne peut pas être partagé entre plusieurs instances** (donc pas de
scaling horizontal avec cette option). Il faut un plan payant.

### Option B — S3 / Cloudflare R2 (recommandé pour scaler)
1. Crée un bucket + une clé d'accès (R2 : **Object Read & Write**).
2. Rends le bucket public **ou** utilise un domaine CDN, et renseigne
   `S3_PUBLIC_BASE_URL`.
3. Sur Render, mets `STORAGE_DRIVER=s3` + les variables `S3_*`.
4. Le conteneur reste **entièrement éphémère** ; après chaque rendu/upload, le
   MP4 et sa vignette sont poussés sur le bucket et l'API renvoie l'URL publique.

> **Implémentation sans SDK** : la signature **AWS SigV4** est calculée avec
> `crypto` natif (aucune dépendance lourde). Compatible **AWS S3, Cloudflare R2,
> MinIO, Backblaze B2**. Si l'upload échoue (bucket momentanément injoignable),
> l'API **dégrade proprement** vers l'URL locale au lieu de planter.

### Comportement du code
- **Intermediaires FFmpeg** (`/tmp/football_engine/work`) : toujours locaux et
  **disposables** (jamais écrits sur le disque persistant → pas de saturation).
- **Sorties finales** (`final_video.mp4`, posters, uploads) : écrites dans
  `PUBLIC_DIR` puis `publish()` → upload S3 si mode `s3`.
- **`GET /health`** renvoie maintenant `{"storage":"local|s3"}` pour vérifier le
  mode actif d'un coup d'œil.
- Au démarrage, les logs indiquent clairement le mode et alertent si le dossier
  éphémère par défaut est utilisé.

### Nettoyage automatique
`MEDIA_RETENTION_HOURS` (ex. `48`) purge périodiquement les fichiers générés
plus vieux que N heures — évite de saturer le disque sur les longs uptimes.


---

## 4. Rappels importants
- Ne commite jamais `.env` / `GEMINI_API_KEY` (déjà dans `.gitignore`).
- `.dockerignore` exclut `node_modules`, `dist`, `.env*`, logs et fichiers
  `uploaded_match_*` de test.
- Pour lancer l'image localement (si Docker est dispo) :
  ```bash
  docker build -t football-cinematic-ai .
  docker run --rm -p 3000:3000 -e GEMINI_API_KEY=xxx football-cinematic-ai
  ```
