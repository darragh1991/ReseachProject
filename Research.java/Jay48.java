package Resarch;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.unsupervised.attribute.Remove;
import weka.classifiers.trees.J48;

public class Jay48 {

	public static void main(String[] args) throws Exception {

		Incremental();

		TrainTestSet();

		FilteringOnTheFly();
	}

	public static void Incremental() throws Exception {
		// source: https://weka.wikispaces.com/Use+Weka+in+your+Java+code
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File("ArffFiles\\3FightAB.arff"));
		Instances structure = loader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);

		J48 j48 = new J48();

		j48.buildClassifier(structure);

		System.out.println("-----------Interface------------");
		System.out.println(j48);
	}

	public static void FilteringOnTheFly() throws Exception {

		System.out.println("\t+++++++++++Predictions++++++++++++\n");

		Instances train = new Instances(new BufferedReader(new FileReader("ArffFiles\\3FightAB.arff")));
		Instances test = new Instances(new BufferedReader(new FileReader("ArffFiles\\3FightABQ.arff")));

		train.setClassIndex(train.numAttributes() - 1);
		test.setClassIndex(test.numAttributes() - 1);
		// filter

		
		J48 j48 = new J48();
		j48.setUnpruned(false);

		FilteredClassifier fc = new FilteredClassifier();
		
		fc.setClassifier(j48);
		// train and make predictions

		fc.buildClassifier(train);

		for (int i = 0; i < test.numInstances(); i++) {
			int predict = (int) fc.classifyInstance(test.instance(i));
			System.out.print("ID: " + test.instance(i).value(0));
			System.out.print("\tactual: " + test.classAttribute().value((int) test.instance(i).classValue()));
			System.out.println("\tpredicted: " + test.classAttribute().value(predict));
		}

	}

	public static void TrainTestSet() throws Exception {

		Instances train = new Instances(new BufferedReader(new FileReader("ArffFiles\\3FightAB.arff")));
		Instances test = new Instances(new BufferedReader(new FileReader("ArffFiles\\3FightABQ.arff")));

		train.setClassIndex(train.numAttributes() - 1);
		test.setClassIndex(test.numAttributes() - 1);

		Classifier j48 = new J48();
		j48.buildClassifier(train);

		Evaluation eval = new Evaluation(train);
		eval.evaluateModel(j48, test);
		System.out.println(eval.toSummaryString("\nResults\n======\n", true));

	}

}
