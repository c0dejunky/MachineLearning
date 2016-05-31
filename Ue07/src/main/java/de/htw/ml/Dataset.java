package de.htw.ml;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

import org.jblas.FloatMatrix;

public class Dataset {
	
	protected Random rnd = new Random(7);
	
	protected FloatMatrix xTrain;
	protected FloatMatrix yTrain;
	
	protected FloatMatrix xTest;
	protected FloatMatrix yTest;
	protected int testDataCount;
	
	protected int[] categories;
	protected int[] categorySizes;
	
	public Dataset() throws IOException {
		
		int predictColumn = 15; // type of apartment
		FloatMatrix data = FloatMatrix.loadCSVFile("german_credit_jblas.csv");
		
		// Liste mit allen Kategorien die es in der predictColumn gibt
		final FloatMatrix outputData = data.getColumn(predictColumn);
		categories = IntStream.range(0, outputData.rows).map(idx -> (int)outputData.data[idx]).distinct().sorted().toArray();
		categorySizes = IntStream.of(categories).sorted().map(v -> (int)outputData.eq(v).sum()).toArray();
		System.out.println("The unique values of y are "+ Arrays.toString(categories)+" and there number of occurrences are "+Arrays.toString(categorySizes));

		// Array mit allen Zeilen die nicht predictColumn sind
		int[] xColumns = IntStream.range(0, data.columns).filter(value -> value != predictColumn).toArray();

		// Ein- und Ausgangsdaten
		FloatMatrix x = data.getColumns(xColumns);
		FloatMatrix y = data.getColumn(predictColumn);

		// min und maximum für alle Spalten
		FloatMatrix xMin = x.columnMins();
		FloatMatrix xMax = x.columnMaxs();
		
		// TODO erstelle ein Trainings- und ein Testset mit 90% und 10% aller Daten
		testDataCount = data.getRows()/10; // 10% Testset
		
		// TODO normalisiere die Datensets
	}

	public FloatMatrix getXTrain() {
		return xTrain;
	}

	public FloatMatrix getYTrain() {
		return yTrain;
	}

	public FloatMatrix getXTest() {
		return xTest;
	}

	public FloatMatrix getYTest() {
		return yTest;
	}

	public int[] getCategories() {
		return categories;
	}

	/**
	 * Bereitet die Trainingsdaten vor. Das Set hat genauso viele Datenpunkte mit gewünschten Kategorie 
	 * wie auch Datenpunkte mit einer anderen Kategorie. Alle Y-Daten sind aber binariziert:
	 *  - gewünschten Kategorie = 1
	 *  - andere Kategorien = 0
	 * 
	 * @param categoryIndex
	 * @return {x,y}
	 */
	public FloatMatrix[] createTrainingsSet(int categoryIndex) {
		int category = categories[categoryIndex];
		int trainingsCategorySize = categorySizes[categoryIndex] - (testDataCount/categories.length);
		
		// TODO finde so viele Indizies von Zeilen in der die Kategorie vorkommt, wie Zeilen mit einer anderen Kategorie
		int[] rowIndizies = new int[] { 1 };
		
		// besorge die gewünschten Datenpunkte und binarisiere die Y-Werte
		return new FloatMatrix[] { xTrain.getRows(rowIndizies), yTrain.getRows(rowIndizies).eq(category) };
	}

	/**
	 * Bereitet das Testset vor. Binariziert die Kategorien:
	 *  - gewünschten Kategorie = 1
	 *  - andere Kategorien = 0
	 *  
	 * @param categoryIndex
	 * @return {x,y}
	 */
	public FloatMatrix[] createTestSet(int categoryIndex) {
		return new FloatMatrix[] { xTest, yTest.eq(categories[categoryIndex]) };
	}
}
