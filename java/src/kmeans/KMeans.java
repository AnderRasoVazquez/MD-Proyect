package kmeans;

import utils.Utils;
import weka.core.*;
import weka.core.stopwords.Null;
import weka.filters.unsupervised.attribute.PrincipalComponents;

import javax.rmi.CORBA.Util;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;

public class KMeans {

    private Instance[] instances;
    private Instance[] centroids;
    private int[] centroidsInstances;
    private Instance[] prevCentroids;
    private boolean[][] belongingBits;
    private int k;
    private double tolerance;
    private DistanceFunction distance;
    private int maxIt;


    public static void main(String[] args) {
        String inputPath = null;
        String outputPath = null;
        int k = 0;
        try {
            inputPath = args[0];
            outputPath = args[1];
            k = Integer.parseInt(args[2]);
        } catch (IndexOutOfBoundsException e) {
            String documentacion = "Este ejecutable realiza el algoritmo de clustering K-Means sobre el conjuto de instancias dado.\n" +
                                    "Tres argumetos esperados:\n" +
                                         "\t1 - Ruta del archivo .arff a leer\n" +
                                         "\t2 - Ruta del archivo .txt a crear. Este archivo contendrá una lista de las instancias y el cluster asignado a cada una\n" +
                                         "\t3 - El número de clusters a formar (k)\n" +
                                    "\nEjemplo: java -jar kMeans.jar /path/to/input/arff /path/to/output/txt k";
            System.out.println(documentacion);
            System.exit(1);
        }


        Instances instances = Utils.loadInstances(inputPath, 0);
        String[] instancesPathList = outputPath.split("/");
        String clustersDir = String.join("/", (String[]) Arrays.asList(instancesPathList).subList(0, instancesPathList.length - 1).toArray(new String[0]));
        clustersDir += "/clusters";
        KMeans kmeans = new KMeans(instances, k, "9-last");
        kmeans.formClusters(outputPath, clustersDir, true);
    }

    public KMeans(Instances pInstances, int pClusters, String pAttrRange) {
        this(pInstances, pClusters, 0.3, pAttrRange);
    }

    public KMeans(Instances pInstances, int pClusters, double pTolerance, String pAttrRange) {
        this.instances = new Instance[pInstances.numInstances()];
        for (int i = 0; i < instances.length; i++) {
            this.instances[i] = pInstances.get(i);
        }
        this.k = Math.min(pClusters, this.instances.length);
        this.centroids = new Instance[this.k];
        this.centroidsInstances = new int[k];
        this.prevCentroids = new Instance[this.k];
        this.belongingBits = new boolean[this.instances.length][this.k];
        this.tolerance = pTolerance;
        Range attrRange = new Range();
        attrRange.setRanges(pAttrRange);
        attrRange.setUpper(pInstances.numAttributes() - 1);
        this.distance = new EuclideanDistance();
        this.distance.setInstances(pInstances);
        this.distance.setAttributeIndices(pAttrRange);
        this.maxIt = 50;
    }

    public void formClusters(String pSavePath) {
        this.formClusters(pSavePath, pSavePath, false);
    }

    public String formClusters(String pInstancesPath, String pClustersDir, boolean pVerbose) {
        // initialize centroids to random instances
        this.initializeCentroids();
        int it = 0;
        // iterate until centroids converge
        while (it < this.maxIt && !this.checkFinished()) {
            it++;
            if (pVerbose)
                System.out.println("It: " + it);
            // update belonging matrix
            this.updateBelongingBits();
            // obtain new centroids
            this.updateCentroids();
            if (pVerbose)
                System.out.println(String.format("Iteration nº %d finished", it));
        }
        // obtain final belonging matrix
        this.updateBelongingBits();
        String centroids = this.getCentroids();
        if (pInstancesPath != null)
            this.saveAssignedInstances(pInstancesPath);
        if (pVerbose)
            System.out.println(this.getCentroids());
        if (pClustersDir != null)
            this.saveClusters(pClustersDir);
        return centroids;
    }

    private void initializeCentroids() {
        Random rng = new Random();
        for (int i = 0; i < this.centroids.length; i++) {
            Instance centroid;
            do {
                centroid = this.instances[rng.nextInt(this.instances.length)];

            } while (Arrays.asList(this.centroids).contains(centroid));
            centroids[i] = centroid;
        }
    }

    private void updateBelongingBits() {
        for (int c = 0; c < this.belongingBits.length; c++) {
            for (int r = 0; r < this.belongingBits[0].length; r++) {
                this.belongingBits[c][r] = false;
            }
        }
        for (int t = 0; t < this.instances.length; t++) {
            double min_diff = 99999;
            int min_i = -1;
            Instance instance = this.instances[t];
            for (int i = 0; i < this.centroids.length; i++) {
                Instance centroid = this.centroids[i];
                double diff = Arrays.equals(instance.toDoubleArray(), centroid.toDoubleArray()) ? 0 : distance.distance(instance, centroid);
                if (diff < min_diff) {
                    min_diff = diff;
                    min_i = i;
                } else if (diff == min_diff && this.centroidsInstances[i] < this.centroidsInstances[min_i]) {
                    min_diff = diff;
                    min_i = i;
                }
            }
            this.belongingBits[t][min_i] = true;
            this.centroidsInstances[min_i]++;
        }
    }

    private void updateCentroids() {
        // save previous centroids for furture comparison
        this.prevCentroids = this.centroids.clone();

        for (int i = 0; i < this.centroids.length; i++) {
            // nº instances that belong to centroid i
            int bits_i = 0;
            // sum of those instances
            Instance instance_sum = null;
            for (int t = 0; t < this.instances.length; t++) {
                if (this.belongingBits[t][i]) {
                    bits_i++;
                    instance_sum = Utils.addInstances(instance_sum, this.instances[t]);
                }
            }
            this.centroids[i] = Utils.divideInstance(instance_sum, bits_i);
            this.centroidsInstances[i] = 0;
        }
    }

    private boolean checkFinished() {
        if (prevCentroids[0] == null) {
            return false;
        }
        for (int i = 0; i < this.centroids.length; i++) {
            Instance centroid = this.centroids[i];
            Instance prev = this.prevCentroids[i];
            if (distance.distance(centroid, prev) > this.tolerance) {
                return false;
            }
        }
        return true;
    }
    
    private String getCentroids() {
        StringBuilder centroids = new StringBuilder();
        centroids.append(String.format("k: %d\n", this.k));
        for (int i = 0; i < this.centroids.length; i++) {
            centroids.append(String.format("CENTROID %d -> %s\n", i, Utils.instanceToString(this.centroids[i], true)));
        }
        return centroids.toString();
    }
    
    private String getClusters() {
        StringBuilder clusters = new StringBuilder();
        clusters.append(String.format("k: %d\n", this.k));
        int attrToPrint = 4;
        for (int i = 0; i < this.centroids.length; i++) {
            clusters.append(String.format("CLUSTER %d\n", i));
            clusters.append(String.format("\tcentroid: %s\n", Utils.instanceToString(this.centroids[i], attrToPrint, true)));
            clusters.append("\tinstances:\n");
            for (int t = 0; t < this.instances.length; t++) {
                if (this.belongingBits[t][i])
                    clusters.append(String.format("\t\t%s\n", Utils.instanceToString(this.instances[t], attrToPrint, true)));
            }
        }
        return clusters.toString();
    }

    private void saveAssignedInstances(String pPath) {
        StringBuilder instances = new StringBuilder();
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(pPath));
            PrintWriter out = new PrintWriter(bw);
            for (int t = 0; t < this.instances.length; t++) {
                for (int i = 0; i < this.centroids.length; i++) {
                    if (this.belongingBits[t][i]) {
                        out.println(String.format("INSTANCE %d -> CLUSTER %d // %s\n",
                                t, i, Utils.instanceToString(this.instances[t], true)));
                        break;
                    }
                }
            }
            out.close();
            bw.close();
        } catch (IOException e) {
            Utils.printlnError(String.format("Error al escribir en %s", pPath));
            e.printStackTrace();
        }
    }

    private void saveClusters(String pDir) {
        Instances instances = new Instances(this.instances[0].dataset());
        instances.setClassIndex(3);
        instances = Utils.filterPCA(instances, 3, 2);
        instances.renameAttribute(0, "x");
        instances.renameAttribute(1, "y");
        File dirFile = new File(pDir);
        if (!dirFile.exists()) {
            dirFile.mkdir();
        }
        for (int i = 0; i < this.centroids.length; i++) {
            Instances cluster = new Instances(this.centroids[i].dataset());
            cluster.delete();
            cluster.add(this.centroids[i]);
            for (int t = 0; t < this.instances.length; t++) {
                if (this.belongingBits[t][i]) {
                    cluster.add(instances.get(t));
                }
            }

            String path = pDir;
            if (path.endsWith("/"))
                path = path.substring(0, path.length() - 1);
            path += String.format("/cluster%d.csv", i);
            System.out.println(path);
            Utils.saveInstancesCSV(cluster, path);
        }
    }

    private void saveClustersBad(String pDir) {
        File dirFile = new File(pDir);
        if (!dirFile.exists()) {
            dirFile.mkdir();
        }
        for (int i = 0; i < this.centroids.length; i++) {
            Instances cluster = new Instances(this.centroids[i].dataset());
            cluster.delete();
            cluster.add(this.centroids[i]);
            for (int t = 0; t < this.instances.length; t++) {
                if (this.belongingBits[t][i]) {
                    cluster.add(this.instances[t]);
                }
            }
            try {
                cluster.setClassIndex(3);
                cluster = Utils.filterPCA(cluster, 3, 2);
                cluster.renameAttribute(0, "x");
                cluster.renameAttribute(1, "y");
            } catch (NullPointerException e) {
                System.out.println("ERROR");
                continue;
            }

            String path = pDir;
            if (path.endsWith("/"))
                path = path.substring(0, path.length() - 1);
            path += String.format("/cluster%d.csv", i);
            System.out.println(path);
            Utils.saveInstancesCSV(cluster, path);
        }
    }

}
