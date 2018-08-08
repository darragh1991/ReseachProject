package NaiveBayes;

public class Tester {

	
	public static void main(String[] args) throws Exception {
		NaivBayes nb = new NaivBayes();
		
		nb.Incremental();
		nb.FilteringOnTheFly();
		nb.TrainTestSet();
		

	}

}
