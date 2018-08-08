package J48;

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


	public Jay48() {
		
	}

	public void Incremental() throws Exception {

		ArffLoader loader = new ArffLoader(); // creating new ArffLoader reference object
		loader.setFile(new File("ArffFiles\\3FightAB.arff")); // using that reference to call the setFile method and
		// then set that to the my desired file
		Instances structure = loader.getStructure(); // setting a new object i.e. structure to the arffLoader method
		// getStructure using the object reference loader
		structure.setClassIndex(structure.numAttributes() - 1); // using the reference object created and set to the
		// getStructure to call the setClassIndex method from
		// within the Instances class and the using that
		// referencing object to call the numAttributes from the
		// Instances class and setting it to -1 to correspond
		// with the required dataset

		J48 j48 = new J48(); // creating a J48 object and setting it to the empty constructor within the j48
								// class

		j48.buildClassifier(structure); // calling the method from NaiveBayes and passing into it the object created and
		// set above

		System.out.println("-----------Interface------------");
		System.out.println(j48);
	}

	public void FilteringOnTheFly() throws Exception {

		System.out.println("\t+++++++++++Predictions++++++++++++\n");

		Instances train = new Instances(new BufferedReader(new FileReader("ArffFiles\\3FightAB.arff")));
		Instances test = new Instances(new BufferedReader(new FileReader("ArffFiles\\3FightABQ.arff")));
		// creating two objects from weka.core.Instances and setting it to the Instances
		// constructor and passing in the file from FileReader constructor via the
		// bufferedReader constructor

		train.setClassIndex(train.numAttributes() - 1);
		test.setClassIndex(test.numAttributes() - 1);
		// using the referencing object to call method setClassIndex and using it
		// objects to set the numAttributes to -1 to match dataset

		J48 j48 = new J48(); // creating a J48 object and setting it to the empty constructor within the j48
								// class
		j48.setUnpruned(false); // using the object to call the setUnpruned method and setting it to false

		FilteredClassifier fc = new FilteredClassifier(); // creating FilteredClassifier object

		fc.setClassifier(j48); // using the filteredClassifier object to call the setClassifier and passes in
		// the NaiveBayes object

		fc.buildClassifier(train); // using the filteredClassifier object to call the buildClassifier and passes in
		// the train object created and set above

		for (int i = 0; i < test.numInstances(); i++) { // iterating through the test object set above
			int predict = (int) fc.classifyInstance(test.instance(i)); // creating a int predict and casting it to
			// FilteredClassifier object to call an method
			// and then pass instances object test and pass
			// into it the iterator i
			System.out.print("ID: " + test.instance(i).value(0)); // printing out an ID reference number and the
			// predictions and the actual
			System.out.print("\tactual: " + test.classAttribute().value((int) test.instance(i).classValue()));
			System.out.println("\tpredicted: " + test.classAttribute().value(predict));
		}

	}

	public void TrainTestSet() throws Exception {

		Instances train = new Instances(new BufferedReader(new FileReader("ArffFiles\\3FightAB.arff")));
		Instances test = new Instances(new BufferedReader(new FileReader("ArffFiles\\3FightABQ.arff")));
		// creating two objects from weka.core.Instances and setting it to the Instances
		// constructor and passing in the file from FileReader constructor via the
		// bufferedReader constructor
		train.setClassIndex(train.numAttributes() - 1);
		test.setClassIndex(test.numAttributes() - 1);
		// using the referencing object to call method setClassIndex and using it
		// objects to set the numAttributes to -1 to match dataset

		Classifier j48 = new J48(); // creating a classifier object and setting it to the empty J48
		// constructor
		j48.buildClassifier(train); // using the object to call the buildClassifier method and pass in train dataset
		// within its parameters

		Evaluation eval = new Evaluation(train); // creating an Evaluation object and setting it to Evaluation(Instances
		// data) passing in train within its parameters
		eval.evaluateModel(j48, test); // using the object to call the evaluateModel which passes in the classifier
		// object and test dataset
		System.out.println(eval.toSummaryString("\nResults\n======\n", true)); // print out results

	}
	// source: https://weka.wikispaces.com/Use+Weka+in+your+Java+code

}
