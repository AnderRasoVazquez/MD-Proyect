package utils;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.PrincipalComponents;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.security.Principal;

/**
 * Esta clase contiene método estáticos para realizar variedad de funciones genéricas e independientes.
 * Para usar los métodos se puede copiar y pegar esta clase en el paquete deseado.
 * A la hora de introducir nuevos métodos, basta con acutalizar la versión de Utils que esté en el ejercicio (examen/práctica) más reciente.
 * Esta es una buena clase para tener disponible en los exámenes de implementación para ahorrar mucho tiempo.
 */
public class Utils {

    /**
     * Carga las instancias del archivo en pPath.
     * Se establece pClassIndex como el índice de la clase. -1 indica que la clase es el último atributo.
     *
     * @param pPath ruta del archivo
     * @param pClassIndex índice del atributo clase
     * @return el objeto de tipo Instances con las instancias del archivo, null si hay problemas al cargar los datos.
     */
    public static Instances loadInstances(String pPath, int pClassIndex) {
        Instances instances = null;
        try {
            ConverterUtils.DataSource ds = new ConverterUtils.DataSource(pPath);
            instances = ds.getDataSet();
            if (pClassIndex >= 0)
                instances.setClassIndex(pClassIndex);
            else
                instances.setClassIndex(instances.numAttributes() - 1);
        } catch (Exception e) {
            printlnError(String.format("Error al cargar las instancias de %s", pPath));
            e.printStackTrace();
        }
        return instances;
    }

    /**
     * Carga las instancias del archivo en pPath.
     * Se supone que la clase es el último atributo.
     *
     * @param pPath ruta del archivo
     * @return el objeto de tipo Instances con las instancias del archivo, null si hay problemas al cargar los datos.
     */
    public static Instances loadInstances(String pPath) {
        return loadInstances(pPath, -1);
    }
    
    /**
     * Carga los datos de un archivo .csv en instancias.
     *
     * @param pPath ruta del archivo a leer
     * @return objeto Instances con las instancias del .csv
     */
    public static Instances loadInstancesCSV(String pPath) {
        CSVLoader csvLoader = new CSVLoader();
        Instances instances;
        try {
            csvLoader.setSource(new File(pPath));
            instances = csvLoader.getDataSet();
        } catch (IOException e) {
            utils.Utils.printlnError(e.getMessage());
            instances = null;
        }
        return instances;
    }

    /**
     * Guarda las instancias pInstances en formato .arff en el archivo pPath.
     *
     * @param pInstances
     * @param pPath
     */
    public static void saveInstances(Instances pInstances, String pPath) {
        try {
            ArffSaver arffSaver = new ArffSaver();
            arffSaver.setInstances(pInstances);
            arffSaver.setFile(new File(pPath));
            arffSaver.writeBatch();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Guarda las instancias pInstances en formato .csv en el archivo pPath.
     *
     * @param pInstances
     * @param pPath
     */
    public static void saveInstancesCSV(Instances pInstances, String pPath) {
        try {
            CSVSaver csvSaver = new CSVSaver();
            csvSaver.setInstances(pInstances);
            csvSaver.setFile(new File(pPath));
            csvSaver.writeBatch();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Obtiene una representacion escrita de todos los valores de la instancia dada.
     * @param pInstance instancia a escribir.
     * @param pSparse flag sparse. Si true, los atributos numéricos con valor 0 se omiten.
     * @return String con los valores de todos los atributos separados por comas.
     */
    public static String instanceToString(Instance pInstance, boolean pSparse) {
        return instanceToString(pInstance, pInstance.numAttributes(), pSparse);
    }

    /**
     * Obtiene una representacion escrita de los valores de la instancia dada.
     * @param pInstance instancia a escribir.
     * @param pAttributes número de atributos de la instancia a escribir.
     * @param pSparse flag sparse. Si true, los atributos numéricos con valor 0 se omiten.
     * @return String con los valores de los atributos separados por comas.
     */
    public static String instanceToString(Instance pInstance, int pAttributes, boolean pSparse) {
        StringBuilder res = new StringBuilder();
        boolean coma = false;
        for (int i = 0; i < pAttributes; i++) {
            Attribute attr = pInstance.attribute(i);
            if (attr.type() == Attribute.NUMERIC) {
                if (pSparse) {
                    if (pInstance.value(i) != 0.0D) {
                        res.append(String.format("%d %f", i, pInstance.value(i)));
                        coma = true;
                    }
                }
                else {
                    res.append(pInstance.value(i));
                    coma = true;
                }
            } else {
                if (pSparse) {
                    res.append(String.format("%d %s", i, attr.value((int) pInstance.value(i))));
                    coma = true;
                }
                else {
                    res.append(attr.value((int) pInstance.value(i)));
                    coma = true;
                }
            }
            if (coma && i < pAttributes -1) {
                res.append(", ");
                coma = false;
            }
        }
        return res.toString();
    }

    /**
     * Suma las dos instancias dadas y obtiene una nueva instancia resultado.
     * Los atributos numéricos de las insctancias se suman de forma normal.
     * Los atributos no númericos se omiten, y la instancia resultado tendrá los mismos
     * valores en estos atributos que la primera de las instancias.
     * @param pInstanceA primera instancia a sumar.
     * @param pInstanceB seguna instancia a sumar.
     * @return instancia resultado de la suma.
     */
    public static Instance addInstances(Instance pInstanceA, Instance pInstanceB) {
        if (pInstanceA == null)
            return pInstanceB;
        else if (pInstanceB == null)
            return pInstanceA;
        else {
            if (pInstanceA.numAttributes() != pInstanceB.numAttributes())
                return null;
            else {
                Instance res = pInstanceA.copy(pInstanceA.toDoubleArray());
                for (int i = 0; i < pInstanceA.numAttributes(); i++) {
                    // if attribute is numeric
                    if (pInstanceA.attribute(i).type() == Attribute.NUMERIC) {
                        res.setValue(i, pInstanceA.value(i) + pInstanceB.value(i));
                    }
                }
                return res;
            }
        }
    }

    /**
     * Divide todos los atributos numéricos de la instancia por el valor dado.
     * @param pInstance Instancia sobre la que aplicar la división.
     * @param pNum Valor por el que dividir los atributos numéricos.
     * @return Instancia con los valores numéricos divididos.
     */
    public static Instance divideInstance(Instance pInstance, double pNum) {
        Instance res = pInstance.copy(pInstance.toDoubleArray());
        for (int i = 0; i < pInstance.numAttributes(); i++) {
            // if attribute is numeric
            if (pInstance.attribute(i).type() == Attribute.NUMERIC) {
                res.setValue(i, pInstance.value(i) / pNum);
            }
        }
        return res;
    }

    /**
     * Aplica el filtro PCA al conjunto de instancias dado.
     * @param pInstances conjunto de instancias para filtrar.
     * @param pClassIndex índice del atributo clase de las instancias.
     * @param pPCAAttr número de atributos (componentes principales) para obtener.
     * @return conjunto de instancias filtradas.
     */
    public static Instances filterPCA(Instances pInstances, int pClassIndex, int pPCAAttr) {
        Instances instances = null;
        try {
            pInstances.setClassIndex(pClassIndex);
            PrincipalComponents pca = new PrincipalComponents();
            pca.setMaximumAttributes(pPCAAttr);
            pca.setMaximumAttributeNames(1);
            pca.setInputFormat(pInstances);
            instances = Filter.useFilter(pInstances, pca);
        } catch (Exception e) {
            System.out.println(pInstances.numInstances());
            e.printStackTrace();
        }
        return instances;
    }

    /**
     * Escribe el texto pTexto en el archivo en la ruta pPath.
     *
     * @param pText texto a escribir
     * @param pPath ruta del archivo
     */
    public static void writeToFile(String pText, String pPath) {
        try {
            BufferedWriter bf = new BufferedWriter(new FileWriter(pPath));
            bf.write(pText);
            bf.close();
        } catch (IOException e) {
            printlnError(String.format("Error al escribir en %s", pPath));
            e.printStackTrace();
        }
    }

    /**
     * Escribe por consola el texto pTexto en color rojo.
     * SIEMPRE incluye salto de línea.
     *
     * @param pText texto a escribir
     */
    public static void printlnError(String pText) {
        printError(String.format("%s\n", pText));
    }

    /**
     * Escribe por consola el texto pTexto en color rojo.
     * NO incluye salto de línea.
     *
     * @param pText texto a escribir
     */
    public static void printError(String pText) {
        System.out.print(String.format("\33[31m%s\33[0m", pText));
    }

    /**
     * Escribe por consola el texto pTexto en color naranja/marrón.
     * SIEMPRE incluye salto de línea.
     *
     * @param pText texto a escribir
     */
    public static void printlnWarning(String pText) {
        printWarning(String.format("%s\n", pText));
    }

    /**
     * Escribe por consola el texto pTexto en color naranja/marrón.
     * NO incluye salto de línea.
     *
     * @param pText texto a escribir
     */
    public static void printWarning(String pText) {
        System.out.print(String.format("\33[33m%s\33[0m", pText));
    }

}
