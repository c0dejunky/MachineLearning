package de.htw.ml;

public class TestDataSet {

	public static void main(String[] args) {
		DataSet d = new DataSet("german_credit_jblas.csv");
		d.splitData(0.9f);

	}

}
