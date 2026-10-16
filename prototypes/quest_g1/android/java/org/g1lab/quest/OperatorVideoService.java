package org.g1lab.quest;

import android.app.*;
import android.content.*;
import android.content.pm.ServiceInfo;
import android.hardware.display.*;
import android.media.*;
import android.media.projection.*;
import android.net.Uri;
import android.os.*;
import android.provider.MediaStore;
import android.util.Log;
import android.view.Surface;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/** Headset compositor -> hardware H.264 -> pending MediaStore MP4. No audio. */
public final class OperatorVideoService extends Service {
    public static volatile int state = 0; // idle, consent, starting, recording, stopping
    public static volatile String message = "MP4 / 1024x1024 / 1:1 / 30 кадров/с / без звука";
    private static final String CHANNEL = "operator-video";
    private static final String STOP = "org.g1lab.quest.STOP_OPERATOR_VIDEO";
    private MediaProjection projection;
    // Square compositor target: preserve more vertical view for hands/racket.
    private static final int VIDEO_WIDTH = 1024, VIDEO_HEIGHT = 1024;
    private VirtualDisplay display;
    private MediaRecorder recorder;
    private Surface surface;
    private ParcelFileDescriptor output;
    private Uri uri;
    private boolean started, closing;
    private String failure;
    private final MediaProjection.Callback callback = new MediaProjection.Callback() {
        @Override public void onStop() { finishCapture(null); }
    };
    static void fail(String text, Exception error) {
        Log.e("G1Video", text, error);
        message = text + ". Повторите запуск";
        state = 0;
    }
    @Override public IBinder onBind(Intent intent) { return null; }
    @Override public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null && STOP.equals(intent.getAction())) {
            finishCapture(null);
            return START_NOT_STICKY;
        }
        if (recorder != null || closing) return START_NOT_STICKY;
        try {
            if (intent == null) throw new IllegalStateException("Missing permission result");
            NotificationManager notifications = getSystemService(NotificationManager.class);
            notifications.createNotificationChannel(new NotificationChannel(CHANNEL,
                "Видео оператора", NotificationManager.IMPORTANCE_LOW));
            PendingIntent stop = PendingIntent.getService(this, 0,
                new Intent(this, OperatorVideoService.class).setAction(STOP),
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
            Notification notification = new Notification.Builder(this, CHANNEL)
                .setSmallIcon(android.R.drawable.presence_video_online)
                .setContentTitle("Идёт запись видео оператора")
                .setContentText("Вид шлема / остановка также во вкладке «Вид»")
                .setOngoing(true).addAction(new Notification.Action.Builder(null, "Остановить", stop).build())
                .build();
            startForeground(4701, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION);
            Intent permission = intent.getParcelableExtra("resultData");
            projection = getSystemService(MediaProjectionManager.class)
                .getMediaProjection(intent.getIntExtra("resultCode", Activity.RESULT_CANCELED), permission);
            if (projection == null) throw new IllegalStateException("No projection");
            projection.registerCallback(callback, new Handler(getMainLooper()));
            long available = new StatFs(Environment.getExternalStorageDirectory().getPath()).getAvailableBytes();
            if (available < 512L * 1024 * 1024) throw new IllegalStateException("Less than 512 MiB free");
            String name = "operator-" + new SimpleDateFormat("yyyyMMdd-HHmmss-SSS", Locale.US).format(new Date()) + ".mp4";
            ContentValues values = new ContentValues();
            values.put(MediaStore.Video.Media.DISPLAY_NAME, name);
            values.put(MediaStore.Video.Media.MIME_TYPE, "video/mp4");
            values.put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/G1Quest");
            values.put(MediaStore.Video.Media.IS_PENDING, 1);
            uri = getContentResolver().insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values);
            if (uri == null) throw new IllegalStateException("Cannot create video");
            output = getContentResolver().openFileDescriptor(uri, "rw");
            if (output == null) throw new IllegalStateException("Cannot open video");
            recorder = new MediaRecorder();
            recorder.setVideoSource(MediaRecorder.VideoSource.SURFACE);
            recorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
            recorder.setOutputFile(output.getFileDescriptor());
            recorder.setVideoEncoder(MediaRecorder.VideoEncoder.H264);
            recorder.setVideoSize(VIDEO_WIDTH, VIDEO_HEIGHT);
            recorder.setVideoFrameRate(30);
            recorder.setVideoEncodingBitRate(8_000_000);
            recorder.setMaxFileSize(Math.min(4_000_000_000L, available - 256L * 1024 * 1024));
            recorder.setOnInfoListener((mr, what, extra) -> {
                if (what == MediaRecorder.MEDIA_RECORDER_INFO_MAX_FILESIZE_REACHED)
                    finishCapture("Видео сохранено: достигнут лимит файла");
            });
            recorder.setOnErrorListener((mr, what, extra) -> {
                failure = "Ошибка кодирования видео";
                Log.e("G1Video", "Recorder error " + what + "/" + extra);
                finishCapture(null);
            });
            recorder.prepare();
            surface = recorder.getSurface();
            display = projection.createVirtualDisplay("G1OperatorVideo", VIDEO_WIDTH, VIDEO_HEIGHT,
                getResources().getDisplayMetrics().densityDpi,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR, surface, null, null);
            if (display == null) throw new IllegalStateException("No virtual display");
            recorder.start();
            started = true;
            message = "Видео: REC / 1024x1024 / 1:1 / 30 / без звука";
            state = 3;
            Log.i("G1Video", "Recording " + uri);
        } catch (Exception error) {
            Log.e("G1Video", "Capture start failed", error);
            failure = "Не удалось начать видео. Проверьте разрешение и место";
            finishCapture(null);
        }
        return START_NOT_STICKY;
    }
    private void finishCapture(String successMessage) {
        if (closing) return;
        closing = true;
        state = 4;
        message = "Сохранение видео...";
        boolean valid = false;
        try {
            if (started) { recorder.stop(); valid = true; }
        } catch (RuntimeException error) {
            Log.e("G1Video", "No complete video frames", error);
            failure = "Видео не сохранено: нет готовых кадров";
        }
        started = false;
        if (display != null) { display.release(); display = null; }
        if (recorder != null) { recorder.release(); recorder = null; }
        if (surface != null) { surface.release(); surface = null; }
        if (projection != null) {
            projection.unregisterCallback(callback);
            projection.stop(); projection = null;
        }
        try {
            if (valid && output != null) {
                MediaMetadataRetriever metadata = new MediaMetadataRetriever();
                try {
                    metadata.setDataSource(output.getFileDescriptor());
                    String duration = metadata.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION);
                    valid = duration != null && Long.parseLong(duration) > 0;
                } finally { metadata.release(); }
            }
            if (output != null) { output.close(); output = null; }
            if (uri != null) {
                if (valid) {
                    ContentValues published = new ContentValues();
                    published.put(MediaStore.Video.Media.IS_PENDING, 0);
                    if (getContentResolver().update(uri, published, null, null) != 1)
                        throw new IllegalStateException("Publish failed");
                    Log.i("G1Video", "Saved " + uri);
                } else getContentResolver().delete(uri, null, null);
            }
        } catch (Exception error) {
            valid = false;
            Log.e("G1Video", "Finalize failed", error);
            failure = "Ошибка сохранения видео";
            try { if (uri != null) getContentResolver().delete(uri, null, null); }
            catch (RuntimeException ignored) { Log.e("G1Video", "Pending video cleanup failed", ignored); }
        } finally {
            try { if (output != null) output.close(); } catch (Exception ignored) { }
            output = null; uri = null;
        }
        message = valid ? (successMessage == null ? "Видео сохранено: Movies/G1Quest" : successMessage)
            : failure == null ? "Видео не сохранено: нет готовых кадров" : failure;
        stopForeground(STOP_FOREGROUND_REMOVE);
        stopSelf();
    }
    @Override public void onDestroy() { finishCapture(null); state = 0; super.onDestroy(); }
}
