package com.emwit.core;

import android.graphics.Bitmap;
import android.graphics.Point;
import android.graphics.Rect;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.ByteArrayOutputStream;
import java.util.ArrayList;

public class Face {
    static {
        System.loadLibrary("detectandrecognize");
        System.loadLibrary("opencv_java3");
    }

    public class FaceInfo {
        public Rect faceRect;
        public float[] feature128;

        public void setFaceRect(Rect faceRect) {
            this.faceRect = faceRect;
        }

        public void setFeature128(float[] feature128) {
            this.feature128 = feature128;
        }
    }

    private class FaceLandmark {
        public Rect faceRect;
        public ArrayList<Point> landmarks;

        public void setFaceRect(Rect faceRect) {
            this.faceRect = faceRect;
        }

        public void setLandmarks(ArrayList<Point> landmarks) {
            this.landmarks = landmarks;
        }
    }

    public boolean init(String modelPath) {
        return FaceModelInit(modelPath);
    }

    public ArrayList<FaceInfo> detectFaceAndGetFeatureFrom(Bitmap imageBitmap) {
        if (imageBitmap == null) {
            return null;
        }
        ArrayList<FaceLandmark> faceLandmarks = detectFaceFrom(imageBitmap);
        ArrayList<FaceInfo> faceInfos = extractFeature(imageBitmap, faceLandmarks);
        return faceInfos;
    }

    private ArrayList<FaceLandmark> detectFaceFrom(Bitmap imageBitmap) {
        ArrayList<FaceLandmark> result = new ArrayList<>();
        byte[] imageByteArray = bitmapToByteArray(imageBitmap);
        int[] faceRectAndLandmark = FaceDetect(imageByteArray, imageBitmap.getWidth(), imageBitmap.getHeight(), 4);
        int faceNumber = faceRectAndLandmark[0];
        for (int i = 0; i < faceNumber; i++) {
            int left = faceRectAndLandmark[1 + 14 * i];
            int top = faceRectAndLandmark[2 + 14 * i];
            int right = faceRectAndLandmark[3 + 14 * i];
            int bottom = faceRectAndLandmark[4 + 14 * i];
            Rect rect = new Rect(left, top, right, bottom);
            ArrayList<Point> points = new ArrayList<>();
            for (int j = 0; j < 5; j++) {
                points.add(new Point(faceRectAndLandmark[j + 5 + 14 * i], faceRectAndLandmark[j + 10 + 14 * i]));
            }
            FaceLandmark faceRegion = new FaceLandmark();
            faceRegion.setFaceRect(rect);
            faceRegion.setLandmarks(points);
            result.add(faceRegion);
        }
        return result;
    }

    private byte[] bitmapToByteArray(Bitmap imageBitmap) {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        imageBitmap.compress(Bitmap.CompressFormat.PNG, 100, baos);
        return baos.toByteArray();
    }

    private ArrayList<FaceInfo> extractFeature(Bitmap imageBitmap, ArrayList<FaceLandmark> faceLandmarks) {
        ArrayList<FaceInfo> faceInfos = new ArrayList<>();
        for (FaceLandmark faceLandmark : faceLandmarks) {
            FaceInfo faceInfo = new FaceInfo();
            faceInfo.setFaceRect(faceLandmark.faceRect);
            Point leftEyePoint = faceLandmark.landmarks.get(0);
            Point rightEyePoint = faceLandmark.landmarks.get(1);
            Bitmap bitmapTemp = enlargeBitmapForWarpAffine(imageBitmap, faceLandmark.faceRect);
            bitmapTemp = warpAffineBitmap(bitmapTemp, leftEyePoint, rightEyePoint);
            FaceLandmark newFaceLandmark = detectFaceFrom(bitmapTemp).get(0);
            int newLeft = newFaceLandmark.faceRect.left;
            int newTop = newFaceLandmark.faceRect.top;
            int newWidth = newFaceLandmark.faceRect.right - newFaceLandmark.faceRect.left;
            int newHeight = newFaceLandmark.faceRect.bottom - newFaceLandmark.faceRect.top;
            bitmapTemp = Bitmap.createBitmap(bitmapTemp, newLeft, newTop, newWidth, newHeight);
            byte[] byteDataForFeatureExtract = bitmapToByteArray(bitmapTemp);
            float[] faceFeatures = GetFaceFeature(byteDataForFeatureExtract, bitmapTemp.getWidth(), bitmapTemp.getHeight());
            faceInfo.setFeature128(faceFeatures);
            faceInfos.add(faceInfo);
        }
        return faceInfos;
    }

    private Bitmap enlargeBitmapForWarpAffine(Bitmap bitmap, Rect rect) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int widthToEnlarge = (int)(rect.right - rect.left) / 3;
        int heightToEnlarge = (int)(rect.bottom - rect.top) / 3;
        int left = (rect.left > widthToEnlarge) ? (rect.left - widthToEnlarge) : 0;
        int top = (rect.top > heightToEnlarge) ? (rect.top - heightToEnlarge) : 0;
        int right = ((rect.right + widthToEnlarge) < width) ? (rect.right + widthToEnlarge) : width;
        int bottom = ((rect.bottom + heightToEnlarge) < height) ? (rect.bottom + heightToEnlarge) : height;
        return Bitmap.createBitmap(bitmap, left, top, right-left, bottom-top);
    }

    private Bitmap warpAffineBitmap(Bitmap bitmap, Point leftEyePoint, Point rightEyePoint) {
        double eyeCenter_x = (leftEyePoint.x + rightEyePoint.x) * 0.5;
        double eyeCenter_y = (leftEyePoint.y + rightEyePoint.y) * 0.5;

        double differenceOfY = rightEyePoint.y - leftEyePoint.y;
        double differenceOfX = rightEyePoint.x - leftEyePoint.x;
        double angleBetweenTwoEyes = Math.atan2(differenceOfY, differenceOfX) * 180.0 / Math.PI;

        // Calculate the rotate mat form eyeCenter angle and scale. scale = 1.0 means keep the same size
        Mat rotateMat = Imgproc.getRotationMatrix2D(new org.opencv.core.Point(eyeCenter_x, eyeCenter_y), angleBetweenTwoEyes, 1.0);
        Mat imgMat = new Mat();
        Utils.bitmapToMat(bitmap, imgMat);

        // Affine
        Mat imgMatAffined = new Mat();
        Imgproc.warpAffine(imgMat, imgMatAffined, rotateMat, imgMat.size());
        Utils.matToBitmap(imgMatAffined, bitmap);
        return bitmap;
    }

    /**
     * Cosine Similarity
     * @param feature1 [in]
     * @param feature2 [in]
     * @return Cosine Similarity
     */
    public double calSimilarityByFeature(float[] feature1, float[] feature2) {
        if (feature1.length == 0 || feature2.length == 0) {
            return 0;
        }
        double ret = 0.0, mod1 = 0.0, mod2 = 0.0;
        for (int index = 0; index < feature1.length; index ++) {
            ret += feature1[index] * feature2[index];
            mod1 += feature1[index] * feature1[index];
            mod2 += feature2[index] * feature2[index];
        }
        return ret / Math.sqrt(mod1) / Math.sqrt(mod2);
    }

    //============================== JNI =======================================
    /**
     * Initial the face detect and recognition model
     * @param modelPath [in] the path of the model
     * @return true for success
     */
    public native boolean FaceModelInit(String modelPath);

    /**
     * Uninitialize the model
     * @return true for success
     */
    public native boolean FaceModelUnInit();

    /**
     * Detect face of an image, convert image to byte array first.
     * The channel can only be 3 or 4.
     * Return a int array containing the information of all faces, the first element
     * of the array is the number of faces, then every 14 elements followed are face
     * infos.
     * @param imageData [in] the byte array data of image
     * @param width [in] the width of the image
     * @param height [in] the height of the image
     * @param channel [in] the channel of the image, should be 3 or 4
     * @return int array containing information of all faces
     */
    public native int[] FaceDetect(byte[] imageData, int width, int height, int channel);

    /**
     * Get feature from face data
     * @param faceData [in] byte array of face
     * @param width [in] the width of face data
     * @param height [in] the height of face data
     * @return float array of feature
     */
    public native float[] GetFaceFeature(byte[] faceData, int width, int height);
}
