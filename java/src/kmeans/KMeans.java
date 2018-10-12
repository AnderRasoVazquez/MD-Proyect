package kmeans;

import utils.Utils;
import weka.core.*;

import java.util.Arrays;
import java.util.Enumeration;
import java.util.Random;

public class KMeans {

    private Instance[] instances;
    private Instance[] centroids;
    private Instance[] prevCentroids;
    private boolean[][] belongingBits;
    private int k;
    private double tolerance;
    private DistanceFunction distance;
    private int maxIt;


    public static void main(String[] args) {
//        Instances instances = Utils.loadInstances("/home/david/Escritorio/test.arff", 0);
        Instances instances = Utils.loadInstances("/media/david/data/Shared/Documentos/Universidad/4º/Minería de Datos/Proyecto/files/verbal_autopsies_raw_tfidf.arff", 0);
        System.out.println(instances.numInstances());
        KMeans kmeans = new KMeans(instances, 2);
        kmeans.formClusters();
    }

    private void test() {
        Instances instances = Utils.loadInstances("/home/david/Escritorio/test.arff", 0);
        for (int i = 0; i < instances.numAttributes(); i++) {
            Attribute attr = instances.attribute(i);
            Enumeration<Object> vals = attr.enumerateValues();
            while (vals != null && vals.hasMoreElements()) {
                System.out.print(vals.nextElement());
                if (vals.hasMoreElements())
                    System.out.print(", ");
                else
                    System.out.println();

            }
        }
        for (int i = 0; i < instances.numInstances(); i++) {
            Instance ins = instances.get(i);
            for (int j = 0; j < ins.numAttributes(); j++) {
                System.out.print(ins.value(j));
                if (j != ins.numAttributes() - 1)
                    System.out.print(", ");
                else
                    System.out.println();
            }
        }
        System.out.println("\nDstances");
        EuclideanDistance distance = new EuclideanDistance();
        distance.setInstances(instances);
        for (int i = 0; i < instances.numInstances(); i++) {
            for (int j = 0; j < instances.numInstances(); j++) {
                Instance insA = instances.get(i);
                Instance insB = instances.get(j);
                System.out.println(i + " - " + j + " -> " + distance.distance(insA, insB));
            }
        }
        System.out.println("\nSums");
        for (int i = 0; i < instances.numInstances(); i++) {
            for (int j = 0; j < instances.numInstances(); j++) {
                Instance insA = instances.get(i);
                Instance insB = instances.get(j);
                System.out.println(i + " - " + j + " -> " + KMeans.addInstances(insA, insB));
            }
        }
    }

    public KMeans(Instances pInstances, int pClusters) {
        this(pInstances, pClusters, 0.1);
    }

    public KMeans(Instances pInstances, int pClusters, double pTolerance) {
        this.instances = new Instance[pInstances.numInstances()];
        for (int i = 0; i < instances.length; i++) {
            this.instances[i] = pInstances.get(i);
        }
        this.k = Math.min(pClusters, this.instances.length);
        this.centroids = new Instance[this.k];
        this.prevCentroids = new Instance[this.k];
        this.belongingBits = new boolean[this.instances.length][this.k];
        this.tolerance = pTolerance;
        this.distance = new EuclideanDistance();
        this.distance.setInstances(pInstances);
        this.maxIt = 100;
    }

    public void formClusters() {
        // initialize centroids to random instances
        this.initializeCentroids();
        int it = 0;
        // iterate until centroids converge
        while (it < this.maxIt && !this.checkFinished()) {
            it++;
            // update belonging matrix
            this.updateBelongingBits();
            System.out.println("It: " + it);
            this.print_status();
            // obtain new centroids
            this.updateCentroids();
            System.out.print("updated centroids");
        }
        // obtain final belonging matrix
        this.updateBelongingBits();
        this.print_status();
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
                    instance_sum = addInstances(instance_sum, this.instances[t]);
                }
            }
            this.centroids[i] = divideInstance(instance_sum, bits_i);
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
            for (int i = 0; i < this.centroids.length; i++) {
                double diff = this.distance.distance(this.instances[t], this.centroids[i]);
                if (diff < min_diff) {
                    min_diff = diff;
                    min_i = i;
                }
            }
            this.belongingBits[t][min_i] = true;
        }
    }

    private boolean checkFinished() {
        boolean finished = true;
        for (int i = 0; i < this.centroids.length; i++) {
            Instance centroid = this.centroids[i];
            Instance prev = this.prevCentroids[i];
            finished = prev != null && distance.distance(centroid, prev) < this.tolerance;
        }
        return finished;
    }

    private static Instance addInstances(Instance pInstanceA, Instance pInstanceB) {
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

    private static Instance divideInstance(Instance pInstance, double pNum) {
        Instance res = pInstance.copy(pInstance.toDoubleArray());
        for (int i = 0; i < pInstance.numAttributes(); i++) {
            // if attribute is numeric
            if (pInstance.attribute(i).type() == Attribute.NUMERIC) {
                res.setValue(i, pInstance.value(i) / pNum);
            }
        }
        return res;
    }

    private void print_status() {
        System.out.println(String.format("k: %d", this.k));
        System.out.println("Instances: ");
        for (int t = 0; t < this.instances.length; t++) {
            System.out.println(String.format("\tt: %d", t));
            for (int i = 0; i < this.centroids.length; i++) {
                System.out.println(String.format("\t\ti: %d -> %s", i, this.belongingBits[t][i]));
            }
        }
    }

}
