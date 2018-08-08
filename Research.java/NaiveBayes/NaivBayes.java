package NaiveBayes;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class NaivBayes {
	
	public NaivBayes() {
		
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

		NaiveBayes nb = new NaiveBayes(); // creating a referencing object to call the NaiveBayes class

		nb.buildClassifier(structure); // calling the method from NaiveBayes and passing into it the object created and
										// set above
		Instance current; // creating object to reference weka.core.Inastance

		while ((current = loader.getNextInstance(structure)) != null) // setting it to arrfLoader.getNextInstance and
																		// passing in the object structure while current
																		// is not equal to null
			nb.updateClassifier(current); // passes in the reference object into the updateClassifier using the
											// referencing the object

		System.out.println("-----------Interface------------");
		System.out.println(nb);
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

		NaiveBayes nb = new NaiveBayes(); // creating NaiveBayes object

		FilteredClassifier fc = new FilteredClassifier(); // creating FilteredClassifier object

		fc.setClassifier(nb); // using the filteredClassifier object to call the setClassifier and passes in
								// the NaiveBayes object

		fc.buildClassifier(train); // using the filteredClassifier object to call the buildClassifier and passes in
									// the train object created and set above

		for (int i = 0; i < test.numInstances(); i++) { // iterating through the test object set above
			int predict = (int) fc.classifyInstance(test.instance(i)); // creating a int predict and casting it to
																		// FilteredClassifier object to call an method
																		// and then pass instances object test and pass
																		// into it the iterator i
			int act = (int) test.instance(i).classValue(); // creating a int act and casting it to object test and pass
															// into it the iterator i and calling the method value
			System.out.print("ID: " + test.instance(i).value(0)); // printing out an ID reference number and the
																	// predictions and the actual
			System.out.print("\tactual: " + test.classAttribute().value(act));
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

		Classifier nb = new NaiveBayes(); // creating a classifier object and setting it to the empty NaiveBayes
											// constructor
		nb.buildClassifier(train); // using the object to call the buildClassifier method and pass in train dataset
									// within its parameters

		Evaluation eval = new Evaluation(train); // creating an Evaluation object and setting it to Evaluation(Instances
													// data) passing in train within its parameters
		eval.evaluateModel(nb, test); // using the object to call the evaluateModel which passes in the classifier
										// object and test dataset
		System.out.println(eval.toSummaryString("\nResults\n======\n", false)); // print out results

	}

	// source: https://weka.wikispaces.com/Use+Weka+in+your+Java+code
}
