package arff_to_tfidf;

import utils.Utils;
import weka.core.Instances;
import weka.core.stemmers.IteratedLovinsStemmer;
import weka.core.stemmers.LovinsStemmer;
import weka.core.stopwords.Rainbow;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.File;

public class AutopsiesArffToTFIDF {

    public static void main(String[] args) {
        String inputPath = null;
        String outputPath = null;
        try {
            inputPath = args[0];
            outputPath = args[1];
        } catch (IndexOutOfBoundsException e) {
            String documentacion = "Este ejecutable convierte el conjunto de datos de autopsias verbales de formato .csv a .arff.\n" +
                                    "El archivo a convertir debe estar en formato .csv y su primera línea debe ser la cabecera de los valores.\n" +
                                    "Dos argumetos esperados:\n" +
                                         "\t1 - Ruta del archivo .csv a leer\n" +
                                         "\t2 - Ruta del archivo .arff a crear\n" +
                                    "\nEjemplo: java -jar autopsiesCSVToArff.jar /path/to/input/csv /path/to/output/arff";
            System.out.println(documentacion);
            System.exit(1);
        }
        autopsiesArffToTFIDF(inputPath, outputPath);
    }


    private static void autopsiesArffToTFIDF(String pInputPath, String pOutputPath) {
        Instances instances = Utils.loadInstances(pInputPath, 3);
        StringToWordVector stringToWordVector = new StringToWordVector(5000);
        String diccPath = new File(pOutputPath).getParentFile().getAbsolutePath() + "/dictionary.txt";
        if (instances != null) {
            try {
                instances.renameAttribute(0, "attr_" + instances.attribute(0).name());
                instances.renameAttribute(1, "attr_" + instances.attribute(1).name());
                instances.renameAttribute(2, "attr_" + instances.attribute(2).name());
                instances.renameAttribute(4, "attr_" + instances.attribute(4).name());
                stringToWordVector.setLowerCaseTokens(true);
                stringToWordVector.setOutputWordCounts(true);
                stringToWordVector.setDictionaryFileToSaveTo(new File(diccPath));
                stringToWordVector.setIDFTransform(true);
                stringToWordVector.setStopwordsHandler(new Rainbow());
                stringToWordVector.setStemmer(new LovinsStemmer());
//                stringToWordVector.setStemmer(new IteratedLovinsStemmer());
                String relationName = instances.relationName();
                stringToWordVector.setInputFormat(instances);
                instances = Filter.useFilter(instances, stringToWordVector);
                instances.setRelationName(relationName);
            } catch (Exception e) {
                e.printStackTrace();
                System.exit(1);
            }

            Utils.saveInstances(instances, pOutputPath);
            System.out.println(String.format("Conversión completa. Nuevo archivo: %s", pOutputPath));
        }

    }

}
