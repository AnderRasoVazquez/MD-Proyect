package arff_to_tfidf;

import utils.Utils;
import weka.core.Instances;
import weka.core.stemmers.IteratedLovinsStemmer;
import weka.core.stemmers.LovinsStemmer;
import weka.core.stemmers.SnowballStemmer;
import weka.core.stopwords.Rainbow;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.File;

public class AutopsiesArffToTFIDF {

    public static void main(String[] args) {
        String inputPath = null;
        String outputPath = null;
        int wordsToKeep = 0;
        try {
            inputPath = args[0];
            outputPath = args[1];
            wordsToKeep = Integer.parseInt(args[3]);
        } catch (IndexOutOfBoundsException | NumberFormatException e) {
            String documentacion = "Este ejecutable aplica el filtro StringToWordVector al atributo de texto de las autopsias verbales.\n" +
                                    "Dos argumetos esperados:\n" +
                                         "\t1 - Ruta del archivo .arff a leer\n" +
                                         "\t2 - Ruta del archivo .arff a crear\n" +
                                         "\t3 - Palabras para quedarse en la transición de String a WordVector\n" +
                                    "\nEjemplo: java -jar autopsiesArffToTFIDF.jar /path/to/input/arff /path/to/output/arff 5000";
            System.out.println(documentacion);
            System.exit(1);
        }
        autopsiesArffToTFIDF(inputPath, outputPath, wordsToKeep, true);
    }

    /**
     * Convierte el atributo de texto de las autopsias a TFIDF.
     *
     * @param pInputPath ruta del archivo .arff con las autopsias.
     * @param pOutputPath ruta del archivo .arff a crear.
     * @param pWordsToKeep número de palabras para mantener en el TFIDF.
     * @param pVerbose si true, se imprime más información por consola.
     */
    private static void autopsiesArffToTFIDF(String pInputPath, String pOutputPath, int pWordsToKeep,  boolean pVerbose) {
        Instances instances = Utils.loadInstances(pInputPath, 3);
        StringToWordVector stringToWordVector = new StringToWordVector(pWordsToKeep);
        String diccPath = new File(pOutputPath).getParentFile().getAbsolutePath() + "/dictionary.txt";
        if (instances != null) {
            try {
                instances.renameAttribute(0, "attr_" + instances.attribute(0).name());
                instances.renameAttribute(1, "attr_" + instances.attribute(1).name());
                instances.renameAttribute(2, "attr_" + instances.attribute(2).name());
                instances.renameAttribute(4, "attr_" + instances.attribute(4).name());
                stringToWordVector.setLowerCaseTokens(true);
                stringToWordVector.setAttributeIndices("last");
                stringToWordVector.setOutputWordCounts(true);
                stringToWordVector.setDictionaryFileToSaveTo(new File(diccPath));
                stringToWordVector.setIDFTransform(true);
//                stringToWordVector.setStopwordsHandler(new Rainbow());
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
            if (pVerbose)
                System.out.println(String.format("Conversión completa. Nuevo archivo: %s", pOutputPath));
        } else {
            if (pVerbose)
                utils.Utils.printlnError("Error en la conversión");
        }

    }

}
