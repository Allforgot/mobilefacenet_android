package com.example.l.mobilefacenet;

import android.graphics.Bitmap;
import android.graphics.Point;
import android.graphics.Rect;

import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.util.ArrayList;

/**
 * Created by L on 2018/6/11.
 */

public class Face {

    public class FaceInfo {
        private Rect rect;
        private ArrayList<Point> landmarkPoints;

        public void setRect(Rect rect) {
            this.rect = rect;
        }

        public Rect getRect() {
            return rect;
        }

        public void setLandmarkPoints(ArrayList<Point> landmarkPoints) {
            this.landmarkPoints = landmarkPoints;
        }

        public ArrayList<Point> getLandmarkPoints() {
            return landmarkPoints;
        }
    }

    /**
     * Detect face in bitmap
     * @param bitmap [in]
     * @param channel [in] the channel of the image
     * @return ArrayList<FaceRegion>, the information of the face, including rectangle and landmarks
     */
    public ArrayList<FaceInfo> faceDetect(Bitmap bitmap, int channel) {
        byte[] imageData = getPixelsRGBA(bitmap);
        int[] detResult = FaceDetect(imageData, bitmap.getWidth(), bitmap.getHeight(), channel);
        int faceNum = detResult[0];
        ArrayList<FaceInfo> result = new ArrayList<>();
        for (int i = 0; i < faceNum; i++) {
            int left = detResult[1 + 14 * i];
            int top = detResult[2 + 14 * i];
            int right = detResult[3 + 14 * i];
            int bottom = detResult[4 + 14 * i];
            Rect rect = new Rect(left, top, right, bottom);
            ArrayList<Point> points = new ArrayList<>();
            for (int j = 0; j < 5; j++) {
                points.add(new Point(detResult[j + 5 + 14 * i], detResult[j + 10 + 14 * i]));
            }
            FaceInfo faceRegion = new FaceInfo();
            faceRegion.setRect(rect);
            faceRegion.setLandmarkPoints(points);
            result.add(faceRegion);
        }

        return result;
    }

    public ArrayList<FaceInfo> faceDetect(Bitmap bitmap) {
        return faceDetect(bitmap, 4);
    }

    private byte[] getPixelsRGBA(Bitmap bitmap) {
        // calculate how many bytes our image consists of
        int bytes = bitmap.getByteCount();
        ByteBuffer buffer = ByteBuffer.allocate(bytes); // Create a new buffer
        bitmap.copyPixelsToBuffer(buffer); // Move the byte data to the buffer

        return buffer.array(); // Get the underlying array containing the
    }

    public native boolean FaceModelInit(String faceDetectionModelPath);

    public native int[] FaceDetect(byte[] imageDate, int imageWidth, int imageHeight, int imageChannel);

    public native boolean FaceModelUnInit();

    public native double FaceRecognize(byte[] faceDate1, int w1, int h1, byte[] faceDate2, int w2, int h2);

    static {
        System.loadLibrary("Face");
    }
}
