# Reference Cinematic Reel

The application now has a dedicated `REFERENCE CINEMATIC REEL` style preset based on the supplied reference clip.

## Editorial language
- 9:16 vertical social-video master
- Immediate visual hook
- Dense montage of real source moments
- Typical shot duration roughly 1–5 seconds
- Alternation of wide, action, detail, close-up and reaction shots
- Tight intelligent reframing around Gemini-selected focal points
- Controlled punch-ins rather than constant zoom
- Selective slow motion for meaningful movement
- Hard cuts as the dominant transition
- One restrained flash reserved for a verified impact when appropriate
- Small white uppercase editorial captions near the lower third
- No opaque subtitle boxes
- Dark cinematic stadium grade with controlled contrast and restrained saturation
- Subtle vignette, sharpening and film grain
- Strong escalation toward a verified climax, then an emotional real-footage outro

## Safety of the edit engine
The reference determines the editing language only. The renderer never imports the reference video's frames, players, logos or watermarks into the user's result.

Every `source_start` and `source_end` must point to the actual uploaded source video. If Gemini cannot analyze the video, the pipeline fails instead of generating a fake football timeline.

## Master output
- 1080x1920
- 30 fps
- H.264 / AAC
- exactly 64.00 seconds
- fast-start MP4

## Optional music
Set `MUSIC_PATH` to a local MP3/AAC/WAV file on the server to mix background music quietly under the original match audio. Without it, the source audio is preserved.


## AI cinematic generation

AI ENHANCED and AI CINEMATIC modes now run Veo 3.1 on the server. The renderer extracts a real source frame as visual guidance, generates short 9:16 bridge/detail shots, then feeds the resulting MP4 files into the same FFmpeg timeline. If `GEMINI_API_KEY` is unavailable, the app safely renders the verified source footage without synthetic inserts.
