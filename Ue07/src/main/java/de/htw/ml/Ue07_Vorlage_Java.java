package de.htw.ml;

import java.io.IOException;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.layout.HBox;
import javafx.stage.Stage;

public class Ue07_Vorlage_Java extends Application {
	
	private static final int TrainingIterations = 3000;
	private static final float LearnRate = 0.5f;
	
	public static void main(String[] args) throws IOException {
		
		// Daten auslesen
		Dataset dataset = new Dataset();
		int[] uniqueValues = dataset.getCategories();
		
		// Expertensystem
		ExpertSystem expert = new ExpertSystem(TrainingIterations, LearnRate, uniqueValues);
		
		// trainiere die einzelnen Logistischen Regressionen
		expert.train(dataset);
		
		// lasse das ganze System eine Vorhersage treffen für alle Daten
		double predictionRate = expert.test(dataset);
		System.out.printf("overall prediction rate %.2f %n", predictionRate);
		
		plot(expert.getPredictionRates(), expert.getTrainErrors(), uniqueValues);
	}
	
	
	// ---------------------------------------------------------------------------------
	// ------------ Alle Änderungen ab hier geschehen auf eigene Gefahr ----------------
	// ---------------------------------------------------------------------------------
	
	private static float[][] predictionRatesPerLabel;
	private static float[][] trainingsErrorPerLabel;
	private static int[] labels;
	
	/**
	 * Start the application and plot the data
	 * 
	 * @param predictionRates
	 * @param trainingsError
	 * @param uniqueValues
	 */
	public static void plot(float[][] predictionRates, float[][] trainingsError, int[] uniqueValues) {
		predictionRatesPerLabel = predictionRates;
		trainingsErrorPerLabel = trainingsError;
		labels = uniqueValues;
		
		Application.launch(new String[0]);
	}
	
	/**
	 * Draw the plot
	 */	
	@Override public void start(Stage stage) {

		HBox pane = new HBox(10, getPredictionRateChart(), getTrainingsErrorChart());		
		Scene scene = new Scene(pane, 1000, 400);
		
		stage.setTitle("Chart");		
		stage.setScene(scene);
		stage.show();
    }
	
	@SuppressWarnings("unchecked")
	protected LineChart<Number, Number> getTrainingsErrorChart() {
		
		final NumberAxis xAxis = new NumberAxis();
		xAxis.setLabel("iteration");
        final NumberAxis yAxis = new NumberAxis();
        yAxis.setLabel("trainings error");
        
		final LineChart<Number, Number> sc = new LineChart<>(xAxis, yAxis);
		sc.setAnimated(false);
		sc.setCreateSymbols(false);
		
		for (int labelIndex = 0; labelIndex < trainingsErrorPerLabel.length; labelIndex++) {
			float[] predictionRates = trainingsErrorPerLabel[labelIndex];
			if(predictionRates == null) continue;

			XYChart.Series<Number, Number> series = new XYChart.Series<>();
			series.setName("Label "+labels[labelIndex]);
			for (int i = 0; i < predictionRates.length; i++) 
				series.getData().add(new XYChart.Data<Number, Number>(i, predictionRates[i]));			
			sc.getData().addAll(series);
		}	
		return sc;
	}
	
	@SuppressWarnings("unchecked")
	protected LineChart<Number, Number> getPredictionRateChart() {
		
		final NumberAxis xAxis = new NumberAxis();
		xAxis.setLabel("iteration");
        final NumberAxis yAxis = new NumberAxis(0, 100, 10);
        yAxis.setLabel("prediction rate");
        
		final LineChart<Number, Number> sc = new LineChart<>(xAxis, yAxis);
		sc.setAnimated(false);
		sc.setCreateSymbols(false);
		
		for (int labelIndex = 0; labelIndex < predictionRatesPerLabel.length; labelIndex++) {
			float[] predictionRates = predictionRatesPerLabel[labelIndex];
			if(predictionRates == null) continue;
			
			XYChart.Series<Number, Number> series = new XYChart.Series<>();
			series.setName("Label "+labels[labelIndex]);
			for (int i = 0; i < predictionRates.length; i++) 
				series.getData().add(new XYChart.Data<Number, Number>(i, predictionRates[i]));			
			sc.getData().addAll(series);
		}	
		return sc;
	}
}
