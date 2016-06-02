package de.htw.ml;

import java.util.Arrays;

import org.jblas.FloatMatrix;

public class ExpertSystem {

	protected LogisticRegression regression;
	
	protected int[] labels; 		// diese Labels soll das System Vorhersagen können
	protected FloatMatrix[] thetas; // die Gewichte für ein Label
	protected float[][] accuracy;	// die prediction rates während des Trainings
	protected float[][] trainErr;	// die prediction rates während des Trainings

	
	public ExpertSystem(int trainingIterations, float learnRate, int[] labels) {
		this.regression = new LogisticRegression(trainingIterations, learnRate);
		this.thetas = new FloatMatrix[labels.length];
		this.accuracy = new float[labels.length][];
		this.trainErr = new float[labels.length][];		
		this.labels = labels;
	}
	
	public void train(Dataset dataset) {
		
		// trainiere eine logistische Regression für jede Kategory
		for (int i = 0; i < labels.length; i++) {	
			
			// erstelle das Trainings- und Testset für diese Kategory
			FloatMatrix[] xyTrainArray = dataset.createTrainingsSet(i);
			FloatMatrix xTrain = xyTrainArray[0];
			FloatMatrix yTrain = xyTrainArray[1];
			
			// erstelle das Testset für diese Kategory
			FloatMatrix[] xyTestArray = dataset.createTestSet(i);
			FloatMatrix xTest = xyTestArray[0];
			FloatMatrix yTest = xyTestArray[1];
			
			// logging
			float ratio = (yTrain.sum() / yTrain.rows * 100);
			System.out.printf("Train category %d (%.2f%% share with %d elements)\n", labels[i], ratio, yTrain.rows);
			
			// starte das Training
			thetas[i] = regression.train(xTest, yTest, xTrain, yTrain);
			accuracy[i] = regression.getLastPredictionRates();
			trainErr[i] = regression.getLastTrainError();
			System.out.printf("best prediction rate %.2f%%\n\n", (new FloatMatrix(accuracy[i])).max());
		}
	}
		
	/**
	 * 
	 * @param dataset
	 * @return
	 */
	public float test(Dataset dataset) {
		
		FloatMatrix xTest = dataset.getXTest();
		FloatMatrix yTest = dataset.getYTest();

		// die Vorhersagen für jedes Label
		FloatMatrix[] hypothesisArr = Arrays.stream(thetas).map(theta -> LogisticRegression.predict(xTest, theta)).toArray(FloatMatrix[]::new);

		// durchlaufe alle Zeilen
		int correctSum = 0;
		for (int r = 0; r < yTest.getRows(); r++) {
			int expectedLabel = (int)yTest.data[r];
			float hypothesisLabel;
			// finde das Theta welches die stärkste Hypothese hat
			//TODO
			for (int i = 0; i < 3; i++) {
				hypo = hypothesisArr[i].get(r);
				if bestHypo > hypo...
				hypothesisArr = dataset.getCategories()[i];
			}

			// stimmt das Label
			if(expectedLabel == hypothesisLabel)
				correctSum++;
		}
		return (float)correctSum / yTest.getRows();
	}
	
	public float[][] getPredictionRates() {
		return accuracy;
	}
	
	public float[][] getTrainErrors() {
		return trainErr;
	}
}
