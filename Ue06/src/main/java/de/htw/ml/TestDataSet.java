package de.htw.ml;

public class TestDataSet {

	public static void main(String[] args) {
		Dataset d = new Dataset("german_credit_jblas.csv");
		d.splitData(0.9f);

	}

}
