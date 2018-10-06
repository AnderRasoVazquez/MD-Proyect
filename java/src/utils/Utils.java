package utils;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

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
