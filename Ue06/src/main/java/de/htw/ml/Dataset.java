package de.htw.ml;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

import org.jblas.FloatMatrix;
import org.jblas.util.Random;

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
		//split row values from positive and negative creditability
		ArrayList<FloatMatrix> pos= new ArrayList<FloatMatrix>();
		ArrayList<FloatMatrix> neg= new ArrayList<FloatMatrix>();
		for (int i = 0; i < data.rows; i++) {
			if(data.get(i, 0) == 1){
				pos.add(data.getRow(i));
			}else if(data.get(i, 0) == 0){
				neg.add(data.getRow(i));
			}
		}
		
		//get test data dependent of trainPercentage
		int trainSetSize = (int) (data.length * trainingPercentage); //1000 * 0.9 = 900
		int testSetSize = (int) (data.length * (1 - trainingPercentage)); //1000 * (1- 0.9) = 100
		
		//test set mit 50/50 pos und neg.. training set besteht aus dem rest
		
		//generate random indices
		Set<Integer> posRandSet = new HashSet<Integer>();
		while (posRandSet.size() == (testSetSize/2)) {
			posRandSet.add(Random.nextInt(pos.size()));
		}
		Integer[] posRandArray = (Integer[]) posRandSet.toArray();
		
		Set<Integer> negRandSet = new HashSet<Integer>();
		while (negRandSet.size() == (testSetSize/2)) {
			negRandSet.add(Random.nextInt(neg.size()));
		}
		Integer[] negRandArray = (Integer[]) negRandSet.toArray();
		
		//get rows by
		for (int i = 0; i < (testSetSize / 2); i++) {
			test.putRow(i, pos.get(posRandArray[i]));
			pos.remove(posRandArray[i]);
			test.putRow(i++, neg.get(negRandArray[i]));
			neg.remove(negRandArray[i]);
		}
		
	}
}
