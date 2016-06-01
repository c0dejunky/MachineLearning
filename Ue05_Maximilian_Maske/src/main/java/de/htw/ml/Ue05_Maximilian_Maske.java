package de.htw.ml;

import java.io.IOException;

import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.util.Random;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.stage.Stage;

public class Ue05_Maximilian_Maske extends Application {

	public static final String title = "Line Chart";
	public static final String xAxisLabel = "Iteration";
	public static final String yAxisLabel = "RMSE";
	
	private static FloatMatrix cars;
	private static FloatMatrix credit;
	private static FloatMatrix xVals;
	private static FloatMatrix rmseValues;
	
	public static void main(String[] args) throws IOException {
		
		gradientMPG(100, 0.3f);
		
//		gradientCreditAmount(300, 8);
		
	}

	private static void gradientCreditAmount(int iterations, float alpha) throws IOException {
		credit = FloatMatrix.loadCSVFile("german_credit_jblas.csv"); //21 columns
		xVals = credit.getColumns(new int[]{0, 1, 2, 3, 4, 6,7 , 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
		FloatMatrix creditAmount = credit.getColumn(5); //Credit Amount in 6. column
		
		
		FloatMatrix normedValues = normRowValues(xVals);
		FloatMatrix theta = linRegression(iterations, alpha, xVals, creditAmount,  normedValues);
		System.out.println("Best Theta: " + theta);
		System.out.println("Best RMSE: " + rmseValues.get(rmseValues.length - 1));
		
		float[] yVals = rmseValues.getColumn(0).toArray();
		plot(yVals);
	}



	private static void gradientMPG(int iterations, float alpha) throws IOException {
		cars = FloatMatrix.loadCSVFile("cars_jblas.csv");
		xVals = cars.getColumns(new int[]{0, 1, 2, 3, 4, 5});
		
		FloatMatrix mpg = cars.getColumn(6);
		FloatMatrix normedValues = normRowValues(xVals);
		FloatMatrix theta = linRegression(iterations, alpha, xVals, mpg, normedValues);
		System.out.println("Best Theta: " + theta);
		System.out.println("Best RMSE: " + rmseValues.get(rmseValues.length - 1));
		
		float[] yVals = rmseValues.getColumn(0).toArray();
		plot(yVals);
	}

	// ---------------------------------------------------------------------------------
	// ------------ Alle Änderungen ab hier geschehen auf eigene Gefahr ----------------
	// ---------------------------------------------------------------------------------
	
	/**
	 * Equivalent zu linspace in Octave
	 * 
	 * @param lower
	 * @param upper
	 * @param num
	 * @return
	 */
	private static FloatMatrix linspace(float lower, float upper, int num) {
        float[] data = new float[num];
        float step = Math.abs(lower-upper) / (num-1);
        for (int i = 0; i < num; i++)
            data[i] = lower + (step * i);
        data[0] = lower;
        data[data.length-1] = upper;
        return new FloatMatrix(data);
    }
	
	private static float[] dataY;
	
	/**
	 * Startet die eigentliche Applikation
	 * 
	 * @param gdppp
	 * @param lifespan
	 * @param xValues
	 * @param yValues
	 * @param args
	 */
	public static void plot(float[] yValues) {
		dataY = yValues;
		Application.launch(new String[0]);
	}
	
	/**
	 * Zeichnet das Diagram
	 */
	@SuppressWarnings("unchecked")
	@Override public void start(Stage stage) {

		stage.setTitle(title);
		
		final NumberAxis xAxis = new NumberAxis();
		xAxis.setLabel(xAxisLabel);
        final NumberAxis yAxis = new NumberAxis();
        yAxis.setLabel(yAxisLabel);
        
		final LineChart<Number, Number> sc = new LineChart<>(xAxis, yAxis);

		XYChart.Series<Number, Number> series1 = new XYChart.Series<>();
		series1.setName("Data");
		for (int i = 0; i < dataY.length; i++) {
			series1.getData().add(new XYChart.Data<Number, Number>(i, dataY[i]));
		}

		sc.setAnimated(false);
		sc.setCreateSymbols(true);

		sc.getData().addAll(series1);

		Scene scene = new Scene(sc, 500, 400);
		stage.setScene(scene);
		stage.show();
    }
	
	/**
	 * root-mean-square error
	 * rmse = sqrt(sum((y1 .- y2).^2)/size(y2,1));
	 * @param x
	 * @param y
	 * @return
	 */
	public static float getRMSE(FloatMatrix x, FloatMatrix y){
		int size = y.length;
		float rmse;
		
		rmse = (float) Math.sqrt((double)MatrixFunctions.pow(x.sub(y), 2).sum() / size);
//		rmse = x.sub(y).mul(x.sub(y));
//		rmse = MatrixFunctions.sqrt(MatrixFunctions.pow(2, x.sub(y)).div(size));
		
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
	public static FloatMatrix getNewTheta(FloatMatrix x, FloatMatrix y, FloatMatrix theta, float alpha){
		 FloatMatrix hTheta = x.mulRowVector(theta).rowSums(); //x * theta
		 FloatMatrix diff = hTheta.sub(y);
		 FloatMatrix deltaTheta = x.transpose().mulRowVector(diff).rowSums();
		 FloatMatrix deltaThetaD = deltaTheta.mul(alpha / x.length);
		 FloatMatrix thetaNew = theta.sub(deltaThetaD);
		return thetaNew;
	}
	
	public static FloatMatrix norm(FloatMatrix x) {
		float min = x.min();
		float max = x.max();
		FloatMatrix norm = x.sub(min).div(max - min);
		return norm;
	}
	
	public static FloatMatrix denorm(FloatMatrix x, FloatMatrix norm) {
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
	public static FloatMatrix linCombi(FloatMatrix theta, FloatMatrix x) {
		FloatMatrix y = x.mulRowVector(theta).rowSums();
		return y;
	}
	
	/**
	 * norm all values by row from FloatMatrix x
	 * @param x
	 * @return
	 */
	public static FloatMatrix normRowValues(FloatMatrix x) {
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
	public static FloatMatrix linRegression(int iterations, float alpha, FloatMatrix values, FloatMatrix x, FloatMatrix normedValues) {
		Random.seed(7);
		FloatMatrix theta = FloatMatrix.rand(values.columns , 1);
		FloatMatrix normedX = norm(x);
		
		//Initial RMSE
		FloatMatrix prediction = denorm(x, linCombi(theta, normedValues));
		rmseValues = new FloatMatrix(iterations);
		rmseValues.put(0, getRMSE(x, prediction));		
		
		for (int i = 1; i < iterations; i++) {
			theta = getNewTheta(normedValues, normedX, theta, alpha);
			prediction = denorm(x, linCombi(theta, normedValues));
			rmseValues.put(i, getRMSE(x, prediction));
		}
		return theta;
	}
}
