package Resarch;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.unsupervised.attribute.*;

public class Weka {

	public static void main(String[] args) throws Exception {

		Incremental();

		TrainTestSet();

		FilteringOnTheFly();
	}

	public static void Incremental() throws Exception {

		ArffLoader loader = new ArffLoader();
		loader.setFile(new File("ArffFiles\\3FightAB.arff"));
		Instances structure = loader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);

		NaiveBayes nb = new NaiveBayes();

		nb.buildClassifier(structure);
		Instance current;

		while ((current = loader.getNextInstance(structure)) != null)
			nb.updateClassifier(current);

		System.out.println("-----------Interface------------");
		System.out.println(nb);
	}

	public static void FilteringOnTheFly() throws Exception {

		System.out.println("\t+++++++++++Predictions++++++++++++\n");

		Instances train = new Instances(new BufferedReader(new FileReader("ArffFiles\\3FightAB.arff")));
		Instances test = new Instances(new BufferedReader(new FileReader("ArffFiles\\3FightABQ.arff")));

		train.setClassIndex(train.numAttributes() - 1);
		test.setClassIndex(test.numAttributes() - 1);
		// filter

		NaiveBayes nb = new NaiveBayes();

		FilteredClassifier fc = new FilteredClassifier();

		fc.setClassifier(nb);
		// train and make predictions

		fc.buildClassifier(train);

		for (int i = 0; i < test.numInstances(); i++) {
			int predict = (int) fc.classifyInstance(test.instance(i));
			int act = (int) test.instance(i).classValue();
			System.out.print("ID: " + test.instance(i).value(0));
			System.out.print("\tactual: " + test.classAttribute().value(act));
			System.out.println("\tpredicted: " + test.classAttribute().value(predict));

		}

	}

	public static void TrainTestSet() throws Exception {

		Instances train = new Instances(new BufferedReader(new FileReader("ArffFiles\\3FightAB.arff")));
		Instances test = new Instances(new BufferedReader(new FileReader("ArffFiles\\3FightABQ.arff")));

		train.setClassIndex(train.numAttributes() - 1);
		test.setClassIndex(test.numAttributes() - 1);

		Classifier nb = new NaiveBayes();
		nb.buildClassifier(train);

		Evaluation eval = new Evaluation(train);
		eval.evaluateModel(nb, test);
		System.out.println(eval.toSummaryString("\nResults\n======\n", false));

	}

}
