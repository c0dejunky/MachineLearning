package de.htw.ml;

import org.jblas.FloatMatrix;

public class LogisticRegression {
	
	protected int trainingIterations;
	protected float learnRate;
	protected float[] predictionRates;
	protected float[] trainErrors;	
	
	public LogisticRegression(int trainingIterations, float learnRate) {
		this.trainingIterations = trainingIterations;
		this.learnRate = learnRate;
	}

	public FloatMatrix train(FloatMatrix xTest, FloatMatrix yTest, FloatMatrix xTrain, FloatMatrix yTrain) {
		this.predictionRates = new float[trainingIterations];
		this.trainErrors = new float[trainingIterations];
		
		// initializiere die Gewichte
		org.jblas.util.Random.seed(7);
		FloatMatrix theta = FloatMatrix.rand(xTrain.getColumns(), 1);
		
		// aktueller Trainingsfehler
		trainErrors[0] = cost(predict(xTrain, theta), yTrain);

		// aktuelle Prediction Rate
		float bestPredictionRate = predictionRates[0] = predictionRate(predict(xTest, theta), yTest);
		
		// training
		FloatMatrix bestTheta = theta.dup();
		int m = yTrain.rows;
		// Training für die logistische Regression
		for (int iteration = 1; iteration < trainingIterations; iteration++) {
			//Hypothesis
			FloatMatrix hypoTheta = predict(xTrain, theta);
			//Difference
			FloatMatrix diff = hypoTheta.sub(yTrain);
			//Desired Change
			FloatMatrix deltaTheta = xTrain.transpose().mmul(diff);
			//absorption
			deltaTheta = deltaTheta.mul(learnRate/m);
			//update
			theta = theta.sub(deltaTheta);
		}
		bestTheta = theta;
		
		return bestTheta;
	}

	/**
	 * Berechnet eine Prediction für die Eingangsdaten X und den aktuellen Gewichten theta
	 * 
	 * @param x
	 * @param theta
	 * @return
	 */
	public static FloatMatrix predict(FloatMatrix x, FloatMatrix theta) {
		FloatMatrix z = x.mmul(theta);
		FloatMatrix hypoTheta = sigmoidi(z);
		return hypoTheta;
	}


	/**
	 * Berechnet den Trainingsfehler mit der logistischen Kostenfunktion oder den RMSE aus.
	 * 
	 * @param prediction
	 * @param y
	 * @return
	 */
	public static float cost(FloatMatrix prediction, FloatMatrix y) {
        FloatMatrix p = prediction.ge(0.5f);
        return p.sub(y).norm1();
	}

	/**
	 * Berechnet zwischen der Prediktion und den Wunschergebnis Y eine Prediktionsrate aus.
	 * 
	 * @param prediction
	 * @param y
	 * @return
	 */
	public static float predictionRate(FloatMatrix prediction, FloatMatrix y) {
		float error = cost(prediction, y);
        int m = y.rows;
        return (m-error)/m * 100;
	}

	/**
	 * Prediction Rates vom letzten Training
	 * 
	 * @return
	 */
	public float[] getLastPredictionRates() {
		return predictionRates;
	}
	
	/**
	 * Error Rates vom letzten Training
	 * 
	 * @return
	 */
	public float[] getLastTrainError() {
		return trainErrors;
	}
	
	/**
	 * Ersetzt die Werte in der Input Matrix mit der sigmoid Variante.
	 * 
	 * @param input
	 * @return
	 */
	public static FloatMatrix sigmoidi(FloatMatrix input) {
		for (int i = 0; i < input.data.length; i++)
			input.data[i] = (float) (1. / ( 1. + Math.exp(-input.data[i]) ));
		return input;
	}
}