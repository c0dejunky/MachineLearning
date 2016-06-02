package de.htw.ml;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

import org.jblas.FloatMatrix;

public class Dataset {

	private FloatMatrix data;
	private FloatMatrix train;
	private FloatMatrix test;


	public Dataset(String path){
		try {
			data = FloatMatrix.loadCSVFile(path);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void splitData(float trainingPercentage) {
		//FloatMatrix data --> ArrayList
		ArrayList<FloatMatrix> dataList= new ArrayList<FloatMatrix>();
		FloatMatrix[] trainList= new FloatMatrix[data.rows];
		for (int i = 0; i < data.rows; i++) {
			dataList.add(data.getRow(i));
			trainList[i] = data.getRow(i);
		}
		
		//get test data dependent of trainPercentage
		int trainSize = (int) (dataList.size() * trainingPercentage); //1000 * 0.9 = 900
		int testSize = (int) (dataList.size() * (1 - trainingPercentage)); //1000 * (1- 0.9) = 100
		
		//split row values from positive and negative creditability
		ArrayList<FloatMatrix> pos= new ArrayList<FloatMatrix>();
		ArrayList<FloatMatrix> neg= new ArrayList<FloatMatrix>();
		
		//generate random indices
		ArrayList<Integer> randIndices = new ArrayList<Integer>();
		for (int i = 0; i < dataList.size(); i++) {
			randIndices.add(i);
		}
		Collections.shuffle(randIndices);
		
		int i = 0;
		while(pos.size() != (testSize/2)) {
			int rndIndex = randIndices.get(i);
			FloatMatrix row = dataList.get(rndIndex);
			if(row.get(0) == 1) {
				pos.add(row);
				trainList[rndIndex] = null;
			}
		i++;
		}
		
		i=0;
		while(neg.size() != (testSize/2)) {
			int rndIndex = randIndices.get(i);
			FloatMatrix row = dataList.get(rndIndex);
			if(row.get(0) == 0) {
				neg.add(row);
				trainList[rndIndex] = null;
			}
		i++;
		}
		
		//testData to FloatMatrix
		test = new FloatMatrix(testSize, data.columns);
		for (int j = 0; j < pos.size(); j++) {
			FloatMatrix row = pos.get(j);
			test.putRow(j, row);
		}
		for (int j = 0; j < neg.size(); j++) {
			FloatMatrix row = neg.get(j);
			test.putRow(j + testSize / 2, row);
		}
		
		//trainData to FlaotMatrix
		train = new FloatMatrix(trainSize, data.columns);
		int rowIndex = 0;
		for (FloatMatrix row : trainList) {
			if(row != null){
				train.putRow(rowIndex, row);
				rowIndex++;
			}
		}
	}
	
	public FloatMatrix getData() {
		return data;
	}
	
	public void setData(FloatMatrix data) {
		this.data = data;
	}
	
	public FloatMatrix getTrain() {
		return train;
	}
	
	public void setTrain(FloatMatrix train) {
		this.train = train;
	}
	
	public FloatMatrix getTest() {
		return test;
	}
	
	public void setTest(FloatMatrix test) {
		this.test = test;
	}
}
