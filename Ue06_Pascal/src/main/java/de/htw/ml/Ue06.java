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

public class Ue06 extends Application {

	public static final String title = "Line Chart";
	public static final String xAxisLabel = "Iteration";
	public static final String yAxisLabel = "RMSE";
	
	
	public static void main(String[] args) throws IOException {
		//Cars, MPG
//		FloatMatrix cars = FloatMatrix.loadCSVFile("cars_jblas.csv");
//		FloatMatrix carValues = cars.getColumns(new int[]{0, 1, 2, 3, 4, 5});
//		FloatMatrix mpg = cars.getColumn(6);
//		
//		LinearRegression lrCars = new LinearRegression(carValues, mpg, 300, 8);
//		lrCars.printBestRmse();
//		plot(lrCars.rmseValues.toArray());;
		
		//Credit Amount
		FloatMatrix credit = FloatMatrix.loadCSVFile("german_credit_jblas.csv"); //21 columns
		FloatMatrix creditValues = credit.getColumns(new int[]{0, 1, 2, 3, 4, 6,7 , 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
		FloatMatrix creditAmount = credit.getColumn(5); //Credit Amount in 6. column
			
		LinearRegression lrCredit = new LinearRegression(creditValues, creditAmount, 300, 8);
		lrCredit.printBestTheta();
		lrCredit.printBestRmse();
		plot(lrCredit.rmseValues.toArray());
		
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
	

	
	
}
