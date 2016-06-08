package de.htw.ml;

import org.jblas.FloatMatrix;

import java.io.IOException;
import java.util.*;
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

    protected int predictColumn;


    public Dataset() throws IOException {

        predictColumn = 15; // type of apartment
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
        testDataCount = testData.getRows();

        // Ein- und Ausgangsdaten
        xTrain = trainData.getColumns(xColumns);
        yTrain = trainData.getColumn(predictColumn);
        xTest = testData.getColumns(xColumns);
        yTest = testData.getColumn(predictColumn);


        // min und maximum für alle Spalten
        FloatMatrix xMin = data.getColumns(xColumns).columnMins();
        FloatMatrix xMax = data.getColumns(xColumns).columnMaxs();


        // normalisiere die Datensets
        xTrain = xTrain.subRowVector(xMin).divRowVector(xMax.sub(xMin));
        xTest = xTest.subRowVector(xMin).divRowVector(xMax.sub(xMin));

        System.out.print("x: " + xTrain.toString());

    }

    public FloatMatrix[] splitData(float trainingPercentage, FloatMatrix data) {
        //FloatMatrix data --> ArrayList
        ArrayList<FloatMatrix> trainList = new ArrayList<FloatMatrix>();
        Collections.shuffle(trainList, new Random(7));
        for (int i = 0; i < data.rows; i++) {
            trainList.add(data.getRow(i));
        }
        ArrayList<FloatMatrix> testList = new ArrayList<FloatMatrix>();

        //get test data dependent of trainPercentage
        int trainSize = (int) (trainList.size() * trainingPercentage); //1000 * 0.9 = 900
        int testSize = (int) (trainList.size() * (1 - trainingPercentage)); //1000 * (1- 0.9) = 100

        for (int i = 0; i < categories.length; i++) {
            int category = categories[i];
            int pos = 0;

            Iterator<FloatMatrix> itr = trainList.iterator();
            while(itr.hasNext()){
                FloatMatrix currentRow = itr.next();
                if ((int)currentRow.get(predictColumn) == categories[i]){
                    pos++;
                    testList.add(currentRow);
                    itr.remove();
                    // if category has been added 33 times, add new category
                    if(pos >= testSize/categories.length){
                        break;
                    }
                }
            }
        }


        //testData to FloatMatrix
        FloatMatrix test = new FloatMatrix(testList.size(), data.columns);
        int rowIndex = 0;
        for (FloatMatrix row : testList) {
            test.putRow(rowIndex, row);
            rowIndex++;
            }


        //trainData to FlaotMatrix
        FloatMatrix train = new FloatMatrix(trainList.size(), data.columns);
        rowIndex = 0;
        for (FloatMatrix row : trainList) {
            train.putRow(rowIndex, row);
            rowIndex++;
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
        int deltaCategorySize = yTrain.length - trainingsCategorySize;
        int trainingsSetSize = 0;
        if (deltaCategorySize < trainingsCategorySize){
            trainingsSetSize = deltaCategorySize;
        }else{
            trainingsSetSize = trainingsCategorySize;
        }

        // finde so viele Indizies von Zeilen in der die Kategorie vorkommt, wie Zeilen mit einer anderen Kategorie
        //50/50 pos/neg
        int[] rowIndizies = new int[trainingsSetSize];

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
