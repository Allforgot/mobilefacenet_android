package com.emwit.core;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Point;
import android.graphics.Rect;
import android.os.Environment;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;

import static android.content.ContentValues.TAG;

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

    public boolean init(String modelPath) {
        return FaceModelInit(modelPath);
    }

    public ArrayList<FaceInfo> detectFaceAndGetFeatureFrom(Bitmap imageBitmap) {
        if (imageBitmap == null) {
            return null;
        }
        int[] faceLandmarks = detectFaceFrom(imageBitmap);
        if (faceLandmarks == null) {
            return null;
        }
        ArrayList<FaceInfo> faceInfos = extractFeature(imageBitmap, faceLandmarks);
        return faceInfos;
    }

    private int[] detectFaceFrom(Bitmap imageBitmap) {
//        ArrayList<FaceLandmark> result = new ArrayList<>();
        byte[] imageByteArray = getPixelsRGBA(imageBitmap);
        int[] faceRectAndLandmark = FaceDetect(imageByteArray, imageBitmap.getWidth(), imageBitmap.getHeight(), 4);
        return faceRectAndLandmark;
//        if (faceRectAndLandmark.length == 0) {
//            return null;
//        }
//        int faceNumber = faceRectAndLandmark[0];
//        for (int i = 0; i < faceNumber; i++) {
//            int left = faceRectAndLandmark[1 + 14 * i];
//            int top = faceRectAndLandmark[2 + 14 * i];
//            int right = faceRectAndLandmark[3 + 14 * i];
//            int bottom = faceRectAndLandmark[4 + 14 * i];
//            Rect rect = new Rect(left, top, right, bottom);
//            ArrayList<Point> points = new ArrayList<>();
//            for (int j = 0; j < 5; j++) {
//                points.add(new Point(faceRectAndLandmark[j + 5 + 14 * i], faceRectAndLandmark[j + 10 + 14 * i]));
//            }
//            FaceLandmark faceRegion = new FaceLandmark();
//            faceRegion.setFaceRect(rect);
//            faceRegion.setLandmarks(points);
//            result.add(faceRegion);
//        }
//        return result;
    }

    private byte[] getPixelsRGBA(Bitmap image) {
        int bytes = image.getByteCount();
        ByteBuffer buffer = ByteBuffer.allocate(bytes);
        image.copyPixelsToBuffer(buffer);
        byte[] temp = buffer.array();
        return temp;
    }

    private ArrayList<FaceInfo> extractFeature(Bitmap imageBitmap, int[] faceLandmarks) {
        ArrayList<FaceInfo> faceInfos = new ArrayList<>();
        Bitmap bitmapTemp = Bitmap.createBitmap(imageBitmap, 0, 0, imageBitmap.getWidth(), imageBitmap.getHeight());
        int faceNumber = faceLandmarks[0];
        for (int index = 0; index < faceNumber; index ++) {
            FaceInfo faceInfo = new FaceInfo();
            int left = faceLandmarks[1 + 14 * index];
            int top = faceLandmarks[2 + 14 * index];
            int right = faceLandmarks[3 + 14 * index];
            int bottom = faceLandmarks[4 + 14 * index];
            faceInfo.setFaceRect(new Rect(left, top, right, bottom));
            Point leftEyePoint = new Point(faceLandmarks[5 + 14 * index]-left, faceLandmarks[10 + 14 * index]-top);
            Point rightEyePoint = new Point(faceLandmarks[6 + 14 * index]-left, faceLandmarks[11 + 14 * index]-top);
            bitmapTemp = enlargeBitmapForWarpAffine(imageBitmap, faceInfo.faceRect);
            bitmapTemp = warpAffineBitmap(bitmapTemp, leftEyePoint, rightEyePoint);
            int[] newFaceLandmark = detectFaceFrom(bitmapTemp);
            float[] faceFeatures;
            if (newFaceLandmark.length > 1) {
                int newLeft = newFaceLandmark[1];
                int newTop = newFaceLandmark[2];
                int newWidth = newFaceLandmark[3] - newFaceLandmark[1];
                int newHeight = newFaceLandmark[4] - newFaceLandmark[2];
                bitmapTemp = Bitmap.createBitmap(bitmapTemp, newLeft, newTop, newWidth, newHeight);
                byte[] byteDataForFeatureExtract = getPixelsRGBA(bitmapTemp);
                faceFeatures = GetFaceFeature(byteDataForFeatureExtract, bitmapTemp.getWidth(), bitmapTemp.getHeight());
            } else {
                bitmapTemp = Bitmap.createBitmap(imageBitmap, left, top, right-left, bottom-top);
                byte[] byteDataForFeatureExtract = getPixelsRGBA(bitmapTemp);
                faceFeatures = GetFaceFeature(byteDataForFeatureExtract, right-left, bottom-top);
            }
            faceInfo.setFeature128(faceFeatures);
            faceInfos.add(faceInfo);
        }
        if (bitmapTemp != null && !bitmapTemp.isRecycled()) {
            bitmapTemp.recycle();
            bitmapTemp = null;
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

    //TODO bad warp affine, change it
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

    private static void copyModelFileFromAssetsToSD(Context context, String strOutFileName) throws IOException {
//        Log.i(TAG, "start copy file " + strOutFileName);
        String modelBaseDir = Environment.getExternalStorageDirectory().getPath() + File.separator + "faceDetAndRecModel" + File.separator;
        File file = new File(modelBaseDir);
        if (!file.exists()) {
            file.mkdir();
        }

        String tmpFile = modelBaseDir + strOutFileName;
        File f = new File(tmpFile);
        if (f.exists()) {
            Log.i(TAG, "file exists " + strOutFileName);
            return;
        }
        InputStream myInput;
        java.io.OutputStream myOutput = new FileOutputStream(modelBaseDir + strOutFileName);
        myInput = context.getAssets().open(strOutFileName);
        byte[] buffer = new byte[1024];
        int length = myInput.read(buffer);
        while (length > 0) {
            myOutput.write(buffer, 0, length);
            length = myInput.read(buffer);
        }
        myOutput.flush();
        myInput.close();
        myOutput.close();
//        Log.i(TAG, "end copy file " + strOutFileName);
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
