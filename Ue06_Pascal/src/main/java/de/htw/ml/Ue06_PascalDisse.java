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

public class Ue06_PascalDisse extends Application {

	public static final String title = "Line Chart";
	public static final String xAxisLabel = "Iterations";
	public static final String yAxisLabel = "Prediction Rate";
	
	public static void main(String[] args) throws IOException {
		//load files
		FloatMatrix cars = FloatMatrix.loadCSVFile("cars_jblas.csv");
		FloatMatrix credit = FloatMatrix.loadCSVFile("german_credit_jblas.csv");
		
		//set values used for calcualtion
		FloatMatrix values = credit;

		//set number of iterations and alpha
		int iterations = 100;
		float alpha = 0.03f;
		
		//get values from last column
		int lastColumnIndex = values.columns-1;
		FloatMatrix orgY = values.getColumn(lastColumnIndex);
		
		//normalize values
		FloatMatrix normX = getNormValues(values, lastColumnIndex);
		FloatMatrix normY = normalize(orgY);
		
		//do logistic Regression and calc error Rate
		float[] predRates = LogisticRegression.calcLogReg(normX,normY,alpha,iterations);

		//plot prediction Rates
		plot(predRates);

	}
	
	public static FloatMatrix getNormValues(FloatMatrix orgX, int columns){
		FloatMatrix normValues = new FloatMatrix(orgX.rows, columns);
		for(int i = 0; i < columns; i++){
			normValues.putColumn(i, normalize(orgX.getColumn(i)));
		}
		return normValues;
	}
	
	public static FloatMatrix normalize(FloatMatrix v){
		float max = v.max();
		float min = v.min();
		return v.sub(min).div(max-min);
	}
	
	public static FloatMatrix denormalize(FloatMatrix norm, FloatMatrix org){
		float max = org.max();
		float min = org.min();
		return norm.mul(max-min).add(min);
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
	 * @param yValues
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
}
