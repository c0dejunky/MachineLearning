package de.htw.ml;

import org.jblas.FloatMatrix;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import java.util.stream.IntStream;

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
        categories = IntStream.range(0, outputData.rows).map(idx -> (int) outputData.data[idx]).distinct().sorted().toArray();
        categorySizes = IntStream.of(categories).sorted().map(v -> (int) outputData.eq(v).sum()).toArray();
        System.out.println("The unique values of y are " + Arrays.toString(categories) + " and there number of occurrences are " + Arrays.toString(categorySizes));

        // Array mit allen Zeilen die nicht predictColumn sind
        int[] xColumns = IntStream.range(0, data.columns).filter(value -> value != predictColumn).toArray();

        // erstelle ein Trainings- und ein Testset mit 90% und 10% aller Daten
        FloatMatrix[] testTrainData = splitData(0.9f, data);
        System.out.println("test: " + testTrainData[0].toString());
        FloatMatrix testData = testTrainData[0];
        FloatMatrix trainData = testTrainData[1];

        // Ein- und Ausgangsdaten
        xTrain = trainData.getColumns(xColumns);
        yTrain = trainData.getColumn(predictColumn);

        // min und maximum für alle Spalten
        FloatMatrix xMin = xTrain.columnMins();
        FloatMatrix xMax = xTrain.columnMaxs();
        float yMax = yTrain.max();
        float yMin = yTrain.min();


        // normalisiere die Datensets
        xTrain = xTrain.subRowVector(xMin).divRowVector(xMax.sub(xMin));
       // y = y.sub(yMin).div(yMax - yMin);

        System.out.print("x: " + xTrain.toString());

    }

    public FloatMatrix[] splitData(float trainingPercentage, FloatMatrix data) {
        //FloatMatrix data --> ArrayList
        ArrayList<FloatMatrix> dataList = new ArrayList<FloatMatrix>();
        FloatMatrix[] trainList = new FloatMatrix[data.rows];
        for (int i = 0; i < data.rows; i++) {
            dataList.add(data.getRow(i));
            trainList[i] = data.getRow(i);

        }

        //get test data dependent of trainPercentage
        int trainSize = (int) (dataList.size() * trainingPercentage); //1000 * 0.9 = 900
        int testSize = (int) (dataList.size() * (1 - trainingPercentage)); //1000 * (1- 0.9) = 100

        //split row values from positive and negative creditability
        ArrayList<FloatMatrix> pos = new ArrayList<FloatMatrix>();
        ArrayList<FloatMatrix> neg = new ArrayList<FloatMatrix>();

        //generate random indices
        ArrayList<Integer> randIndices = new ArrayList<Integer>();
        for (int i = 0; i < dataList.size(); i++) {
            randIndices.add(i);
        }
        Collections.shuffle(randIndices);

        int i = 0;
        while (pos.size() != (testSize / 2)) {
            int rndIndex = randIndices.get(i);
            FloatMatrix row = dataList.get(rndIndex);
            if (row.get(0) == 1) {
                pos.add(row);
                trainList[rndIndex] = null;
            }
            i++;
        }

        i = 0;
        while (neg.size() != (testSize / 2)) {
            int rndIndex = randIndices.get(i);
            FloatMatrix row = dataList.get(rndIndex);
            if (row.get(0) == 0) {
                neg.add(row);
                trainList[rndIndex] = null;
            }
            i++;
        }

        System.out.println(pos.size());
        System.out.println(neg.size());
        System.out.println(trainList.length);

        //testData to FloatMatrix
        FloatMatrix test = new FloatMatrix(testSize, data.columns);
        for (int j = 0; j < pos.size(); j++) {
            FloatMatrix row = pos.get(j);
            test.putRow(j, row);
        }
        for (int j = 0; j < neg.size(); j++) {
            FloatMatrix row = neg.get(j);
            test.putRow(j + testSize / 2, row);
        }

        //trainData to FlaotMatrix
        FloatMatrix train = new FloatMatrix(trainSize, data.columns);
        int rowIndex = 0;
        for (FloatMatrix row : trainList) {
            if (row != null) {
                train.putRow(rowIndex, row);
                rowIndex++;
            }
        }

        FloatMatrix[] testTrain = new FloatMatrix[]{test, train};

        return testTrain;


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
     * - gewünschten Kategorie = 1
     * - andere Kategorien = 0
     *
     * @param categoryIndex
     * @return {x,y}
     */
    public FloatMatrix[] createTrainingsSet(int categoryIndex) {
        int category = categories[categoryIndex];
        int trainingsCategorySize = categorySizes[categoryIndex] - (testDataCount / categories.length);

        // finde so viele Indizies von Zeilen in der die Kategorie vorkommt, wie Zeilen mit einer anderen Kategorie
        //50/50 pos/neg
        int[] rowIndizies = new int[10];

        int p = 0, n = 0, rowIndex = 0;
        while (p + n < rowIndizies.length) {
            int currentCategory = (int) yTrain.get(rowIndex);
            if(currentCategory == category && p < (rowIndizies.length/2)) {
                rowIndizies[p+n] = rowIndex;
                p++;
            } else if(currentCategory != category && n < (rowIndizies.length/2)){
                rowIndizies[p+n] = rowIndex;
                n++;
            }
            rowIndex++;
        }
        System.out.println("rowIndecies: " + rowIndizies.toString());
        // besorge die gewünschten Datenpunkte und binarisiere die Y-Werte
        return new FloatMatrix[]{xTrain.getRows(rowIndizies), yTrain.getRows(rowIndizies).eq(category)};
    }

    /**
     * Bereitet das Testset vor. Binariziert die Kategorien:
     * - gewünschten Kategorie = 1
     * - andere Kategorien = 0
     *
     * @param categoryIndex
     * @return {x,y}
     */
    public FloatMatrix[] createTestSet(int categoryIndex) {
        return new FloatMatrix[]{xTest, yTest.eq(categories[categoryIndex])};
    }
}
