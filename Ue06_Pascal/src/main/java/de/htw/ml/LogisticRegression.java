package de.htw.ml;

import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.util.Random;

/**
 * Created by Pascal on 01/06/16.
 */
public class LogisticRegression {

    // for testing only
    public static void main(String[] args) {
        FloatMatrix x = new FloatMatrix(new float[][]{
                {0.346f, 0.504f, 0.740f},
                {0.753f, 0.973f, 0.981f},
                {0.585f, 0.539f, 0.417f},
                {0.358f, 0.062f, 0.468f}
        });
        System.out.println(x.toString());

        FloatMatrix y = new FloatMatrix(new float[][]{
                {0.0f},
                {1.0f},
                {0.0f},
                {0.0f}
        });

        System.out.println(y.toString());



        float alpha = 0.3f;
        int m = 4;
        int iterations = 100;


        calcLogReg(x, y, alpha, m, iterations);
    }

    public static FloatMatrix calcLogReg(FloatMatrix x, FloatMatrix y, float v, float alpha, int iterations){
        Random.seed(7);
        FloatMatrix theta = FloatMatrix.rand(x.columns, 1);
        int m = y.rows;
        float[] predRates = new float[iterations];
        for (int i = 0; i < iterations; i++) {
            //Hypothesis
            FloatMatrix hypoTheta = hypothesis(x, theta);
            //Difference
            FloatMatrix diff = hypoTheta.sub(y);
            //Desired Change
            FloatMatrix deltaTheta = x.transpose().mmul(diff);
            //absorption
            deltaTheta = deltaTheta.mul(alpha/m);
            //update
            theta = theta.sub(deltaTheta);

            //binary prediction
            FloatMatrix p = hypoTheta.ge(0.5f);
            //error
            float error = p.sub(y).norm1();
            //prediction Rate
            predRates[i] = (m-error)/m * 100;
        }
        return null;
    }


    private static FloatMatrix binaryPrediction(FloatMatrix hypoTheta){
        FloatMatrix p = new FloatMatrix(hypoTheta.length, 1);
        for (int i = 0; i < hypoTheta.length; i++) {
            if(hypoTheta.get(i) > 0.5){
                p.put(i, 1.0f);
            }else{
                p.put(i,0.f);
            }

        }
        return p;
    }


    private static FloatMatrix hypothesis(FloatMatrix x , FloatMatrix theta){
        FloatMatrix z = x.mmul(theta);
        FloatMatrix hypoTheta = new FloatMatrix(z.rows, 1);
        for (int i = 0; i < hypoTheta.rows; i++) {
            float sigmoid = sigmoid(z.get(i));
            hypoTheta.put(i, sigmoid);
        }
        return hypoTheta;
    }

    private static float sigmoid(float z){
        return (float)(
                1/(1+Math.pow(Math.E, -z)));
    }


    public static FloatMatrix denormalize(FloatMatrix norm, FloatMatrix org) {
        float max = org.max();
        float min = org.min();
        return norm.mul(max - min).add(min);
    }

    public static float rmse(FloatMatrix y, FloatMatrix gleichung) {
        return (float) Math.sqrt((double) MatrixFunctions.pow(y.sub(gleichung), 2).sum() / (gleichung.length));
    }

    public static FloatMatrix linearkombi(FloatMatrix theta, FloatMatrix x) {
        return x.mmul(theta);
    }

    public static FloatMatrix gradient(FloatMatrix x, FloatMatrix y, FloatMatrix theta, float alpha, float m) {
        FloatMatrix hypoTheta = x.mmul(theta);
        FloatMatrix diff = hypoTheta.sub(y);
        FloatMatrix deltaTheta = x.transpose().mmul(diff);
        deltaTheta = deltaTheta.mul(alpha / m);
        return theta.sub(deltaTheta);
    }

    public static Object[] linearRegression(FloatMatrix normX, FloatMatrix normY, FloatMatrix orgY, int iterations, float alpha) {
        Random.seed(7);
        FloatMatrix theta = FloatMatrix.rand(normX.columns, 1);
        int m = normY.rows;
        float[] rmseArray = new float[iterations];

        for (int i = 0; i < iterations; i++) {
            theta = gradient(normX, normY, theta, alpha, m);
            rmseArray[i] = rmse(orgY, denormalize(linearkombi(theta, normX), orgY));
        }
        Object[] thetaRMSE = new Object[2];
        thetaRMSE[0] = theta;
        thetaRMSE[1] = rmseArray;
        return thetaRMSE;
    }
}


