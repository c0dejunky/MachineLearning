package de.htw.ml;

import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.util.Random;

public class LinearRegression {
	
	public FloatMatrix rmseValues;
	public FloatMatrix bestTheta;

	public LinearRegression(FloatMatrix trainValues, FloatMatrix testValues, int iterations, float alpha) {
		FloatMatrix normedValues = normRowValues(trainValues);
		bestTheta = linRegression(iterations, alpha, trainValues, testValues,  normedValues);
	}

	public void printBestRmse() {
		System.out.println("Best RMSE: " + rmseValues.get(rmseValues.length - 1));
	}

	public void printBestTheta() {
		System.out.println("Best Theta: " + bestTheta);
	}
	
	/**
	 * root-mean-square error
	 * rmse = sqrt(sum((y1 .- y2).^2)/size(y2,1));
	 * @param x
	 * @param y
	 * @return
	 */
	public float getRMSE(FloatMatrix x, FloatMatrix y){
		int size = y.length;
		float rmse;
		
		rmse = (float) Math.sqrt((double)MatrixFunctions.pow(x.sub(y), 2).sum() / size);
		
		return rmse;
	}
	
	/**
	 * Neues Theta finden durch erste Ableitung (Gradient) 
	 * @param x
	 * @param y
	 * @param theta
	 * @param alpha
	 * @return
	 */
	public FloatMatrix getNewTheta(FloatMatrix x, FloatMatrix y, FloatMatrix theta, float alpha){
		 FloatMatrix hTheta = x.mmul(theta); //x * theta
		 FloatMatrix diff = hTheta.sub(y);
		 FloatMatrix deltaTheta = x.transpose().mmul(diff);
		 FloatMatrix deltaThetaD = deltaTheta.mul(alpha / x.length);
		 FloatMatrix thetaNew = theta.sub(deltaThetaD);
		return thetaNew;
	}
	
	public FloatMatrix norm(FloatMatrix x) {
		float min = x.min();
		float max = x.max();
		FloatMatrix norm = x.sub(min).div(max - min);
		return norm;
	}
	
	public FloatMatrix denorm(FloatMatrix x, FloatMatrix norm) {
		float min = x.min();
		float max = x.max();
		FloatMatrix denorm = norm.mul(max - min).add(min);
		return denorm;
	}
	
	/**
	 * linear combination
	 * @param theta
	 * @param x
	 * @return
	 */
	public FloatMatrix linCombi(FloatMatrix theta, FloatMatrix x) {
//		FloatMatrix y = x.mulRowVector(theta).rowSums();
		FloatMatrix y = x.mmul(theta);
		return y;
	}
	
	/**
	 * norm all values by row from FloatMatrix x
	 * @param x
	 * @return
	 */
	public FloatMatrix normRowValues(FloatMatrix x) {
		FloatMatrix normedValues = new FloatMatrix(x.rows, x.columns);
		for (int i = 0; i < x.columns; i++) {
			normedValues.putColumn(i, norm(x.getColumn(i)));
		}
		return normedValues;
	}
	
	/**
	 * Gradientenverfarhen: linear regression
	 * @param iterations
	 * @param alpha
	 * @param values
	 * @param normedValues
	 * @return
	 */
	public FloatMatrix linRegression(int iterations, float alpha, FloatMatrix values, FloatMatrix x, FloatMatrix normedValues) {
		Random.seed(7);
		FloatMatrix theta = FloatMatrix.rand(values.columns , 1);
		FloatMatrix normedX = norm(x);
		
		FloatMatrix prediction;
		rmseValues = new FloatMatrix(iterations);	
		
		for (int i = 0; i < iterations; i++) {
			theta = getNewTheta(normedValues, normedX, theta, alpha);
			prediction = denorm(x, linCombi(theta, normedValues));
			rmseValues.put(i, getRMSE(x, prediction));
		}
		return theta;
	}
}
