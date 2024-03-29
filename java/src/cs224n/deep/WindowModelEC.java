package cs224n.deep;
import java.lang.*;
import java.util.*;
import java.io.IOException;

import org.ejml.data.*;
import org.ejml.simple.*;
import org.ejml.ops.*;

import java.text.*;

public class WindowModelEC {

	protected SimpleMatrix L, W1, W2, U, b1, b2;
        //scalar bias
        double b3;
	//
	public int windowSize,wordSize, hiddenSize,hiddenSize2, iterations;
        public double alpha, C, m;

        public boolean checkingGradient;
        private List<Datum> testData;

	public WindowModelEC(int _windowSize, int _hiddenSize, double _lr, List<Datum> _testData){
		windowSize = _windowSize;
                wordSize = FeatureFactory.allVecs.numCols();
                hiddenSize = _hiddenSize;
                hiddenSize2 = hiddenSize/2;
                //K iterations
                iterations = 20;
                alpha = _lr;
                //checkingGradient = true;
                checkingGradient = false;
                C = 0.0001;
                
                testData = _testData;
	}

	/**
	 * Initializes the weights randomly. 
	 */
	public void initWeights(){
		int fanIn = wordSize*windowSize;
                int fanOut = hiddenSize;
                double eInit = Math.sqrt(6.0)/Math.sqrt(fanIn+fanOut);
                W1 = SimpleMatrix.random(fanOut, fanIn, -eInit, eInit, new Random());
                W2 = SimpleMatrix.random(hiddenSize2, fanOut, -eInit, eInit, new Random());
                U = SimpleMatrix.random(hiddenSize2, 1, -eInit, eInit, new Random());
                b1 = SimpleMatrix.random(hiddenSize, 1, -eInit, eInit, new Random());
                b2 = SimpleMatrix.random(hiddenSize2, 1, -eInit, eInit, new Random());
                b3 = (2*Math.random()-1)*eInit;
                L = new SimpleMatrix(FeatureFactory.allVecs);
	}

        private double sigmoid(double z) {
          return (1.0/(1.0+Math.exp(-z)));
        }

        private SimpleMatrix calcTanh(SimpleMatrix m) {
          SimpleMatrix tanhMatrix = new SimpleMatrix(m.numRows(), m.numCols());
          for (int i = 0; i < m.numRows(); i++) {
            for (int j = 0; j < m.numCols(); j++) tanhMatrix.set(i, j, Math.tanh(m.get(i,j)));
          }
          return tanhMatrix;
        }   

        private SimpleMatrix derivTanh(SimpleMatrix tanhMatrix) {
          SimpleMatrix derivMatrix = new SimpleMatrix(tanhMatrix.numRows(), tanhMatrix.numCols());
          for (int i = 0; i < tanhMatrix.numRows(); i++) {
            for (int j = 0; j < tanhMatrix.numCols(); j++) derivMatrix.set(i, j, 1 - Math.tanh(tanhMatrix.get(i,j))*Math.tanh(tanhMatrix.get(i,j)));
          }
          return derivMatrix;
       
        }

        private String[] getWindow(List<Datum> _tD, int pos) {
          String[] window = new String[windowSize];
          int startPos = pos - windowSize/2;
          for (int i = 0; i < windowSize; i++) {
            if (startPos + i < 0) window[i] = "<s>";
            else if (startPos + i >= _tD.size()) window[i] = "</s>";
            else window[i] = _tD.get(startPos+i).word;
          }
          return window;
        }

	/**
	 * Simplest SGD training 
	 */
	public void train(List<Datum> _trainData ){
          System.out.println("Begin training.");
          m = _trainData.size();
          try {
            L.saveToFileCSV("untrainedL.txt");
          } catch (IOException e) {
            System.out.println("Error saving untrained vector.");
            e.printStackTrace();
          }
          for (int it = 0; it < iterations; it++) {
            System.out.println("Iteration " + it + " of " + iterations);
            for (int i = 0; i < _trainData.size(); i++) {
              //System.out.println("Window " + i + " of " + _trainData.size());
	      String[] window = getWindow(_trainData, i);
         
              SimpleMatrix xVec = new SimpleMatrix(windowSize*wordSize, 1);
              for (int j = 0; j < windowSize; j++) {
                int index = 0;
                if (FeatureFactory.wordToNum.containsKey(window[j])) index = FeatureFactory.wordToNum.get(window[j]);
                xVec.insertIntoThis(j*wordSize, 0, L.extractVector(true, index).transpose()); 
              }
              //hSx1
              SimpleMatrix z = W1.mult(xVec).plus(b1);
              SimpleMatrix a = calcTanh(z);
              //hs2x1
              SimpleMatrix z2 = W2.mult(a).plus(b2);
              SimpleMatrix a2 = calcTanh(z2);
              //scalar
              double Utab2 = U.transpose().mult(a2).get(0,0) + b3;
              double h = sigmoid(Utab2);
              
              SimpleMatrix dZ = derivTanh(z);
              SimpleMatrix dZ2 = derivTanh(z2); 

              int y;
              if (_trainData.get(i).label.equals("PERSON")) y = 1;
              else y = 0; 

              //hS2x1
              SimpleMatrix UTdZ2 = dZ2.elementMult(U);
              //hS2x1
              SimpleMatrix W2dZ = W2.mult(dZ);

              //gradients
              double delta2 = h - y;
              SimpleMatrix dJdU = a2.scale(delta2).plus(U.scale(C/m));
              SimpleMatrix dJdW2 = UTdZ2.mult(a.transpose()).scale(delta2).plus(W2.scale(C/m));
              //hs2x1, needs to convert to hsx1
              SimpleMatrix temp1 = UTdZ2.scale(delta2);
              //converts
              SimpleMatrix temp2 = W2.transpose().mult(temp1).elementMult(dZ);  
              SimpleMatrix dJdW1 = temp2.mult(xVec.transpose()).plus(W1.scale(C/m));
              SimpleMatrix dJdb1 = temp2;
              SimpleMatrix dJdb2 = temp1;
              double dJdb3 = delta2;
              SimpleMatrix dJdL = W1.transpose().mult(temp2);

              //if (checkingGradient) gradientCheck(xVec, y);
               
              U = U.minus(dJdU.scale(alpha));
              W1 = W1.minus(dJdW1.scale(alpha));
              W2 = W2.minus(dJdW2.scale(alpha));
              b1 = b1.minus(dJdb1.scale(alpha));
              b2 = b2.minus(dJdb2.scale(alpha));
              b3 = b3-alpha*dJdb3;
              
              for (int j = 0; j < windowSize; j++) {
                int index = 0;
                if (FeatureFactory.wordToNum.containsKey(window[j])) index = FeatureFactory.wordToNum.get(window[j]);
                L.insertIntoThis(index, 0, dJdL.extractMatrix(j*wordSize, (j+1)*wordSize, 0, 1).scale(alpha));
              }
                 
            }
            test(testData);
          }
          System.out.println("Done training.");
          try {
            L.saveToFileCSV("trainedL.txt");
          } catch (IOException e) {
            System.out.println("Error saving trained vector.");
            e.printStackTrace();
          }
        }

        private double getHVal(SimpleMatrix xVec) {
              SimpleMatrix z = W1.mult(xVec).plus(b1);
              SimpleMatrix a = calcTanh(z);
              SimpleMatrix z2 = W2.mult(a).plus(b2);
              SimpleMatrix a2 = calcTanh(z2);
              double Utab2 = U.transpose().mult(a2).get(0,0) + b3;
              double h = sigmoid(Utab2);
          return h;
        }

        private double getCost(double h, int y) {
          return  -y*Math.log(h)-(1-y)*Math.log(1-h)+(C/(2.0*m))*(SpecializedOps.elementSumSq(W2.getMatrix())+SpecializedOps.elementSumSq(W1.getMatrix())+SpecializedOps.elementSumSq(U.getMatrix()));
        }

/*
        private void gradientCheck(SimpleMatrix xVec, int y) {
          //this is NOT pretty but it works...
          double e = Math.pow(10, -4);
          double Udiff = 0;
          double Wdiff = 0;
          double b1diff = 0;
          double b2diff = 0;
          double Ldiff = 0;
          double editMe, h, delta2, costPlus, costMinus, approxDiff;
          SimpleMatrix z, a, dZ, UTdZ;
          //U, W, b1, b2, L
          //U
          for (int i = 0; i < hiddenSize; i++) {
            editMe = U.get(i, 0);
            U.set(i, 0, editMe+e);
            h = getHVal(xVec);
            costPlus = getCost(h,y);
            U.set(i,0, editMe-e);
            h = getHVal(xVec);
            costMinus = getCost(h,y);
            approxDiff = (costPlus-costMinus)/(2.0*e);
            U.set(i,0,editMe);
            z = W.mult(xVec).plus(b1);
            a = calcTanh(z);
            h = getHVal(xVec);
            delta2 = h - y;
            SimpleMatrix dJdU = a.scale(delta2).plus(U.scale(C/m));
            Udiff += Math.pow(Math.abs(dJdU.get(i,0)-approxDiff), 2);
          }
          System.out.println("U difference: " + Udiff);
          //W
          for (int i = 0; i < hiddenSize; i++) {
            editMe = W.get(i);
            W.set(i,editMe+e);
            h = getHVal(xVec);
            costPlus = getCost(h,y);
            W.set(i,editMe-e);
            h = getHVal(xVec);
            costMinus = getCost(h,y);
            approxDiff = (costPlus-costMinus)/(2.0*e);
            W.set(i,editMe);
            z = W.mult(xVec).plus(b1);
            a = calcTanh(z);
            h = getHVal(xVec);
            dZ = derivTanh(z); 
            UTdZ = dZ.elementMult(U);
            delta2 = h - y;
            SimpleMatrix dJdW = UTdZ.mult(xVec.transpose()).scale(delta2).plus(W.scale(C/m));
            Wdiff += Math.pow(Math.abs(dJdW.get(i)-approxDiff), 2);
          }
          System.out.println("W difference: " + Wdiff);
          
          //b1
          for (int i = 0; i < hiddenSize; i++) {
            editMe = b1.get(i);
            b1.set(i,editMe+e);
            h = getHVal(xVec);
            costPlus = getCost(h,y);
            b1.set(i,editMe-e);
            h = getHVal(xVec);
            costMinus = getCost(h,y);
            approxDiff = (costPlus-costMinus)/(2.0*e);
            b1.set(i,editMe);
            z = W.mult(xVec).plus(b1);
            a = calcTanh(z);
            h = getHVal(xVec);
            dZ = derivTanh(z); 
            UTdZ = dZ.elementMult(U);
            delta2 = h - y;
            SimpleMatrix dJdb1 = UTdZ.scale(delta2);
            b1diff += Math.pow(Math.abs(dJdb1.get(i)-approxDiff), 2);
          }
          System.out.println("b1 difference: " + b1diff);
          
          //b2
          editMe = b2;
          b2 = editMe+e;
          h = getHVal(xVec);
          costPlus = getCost(h,y);
          b2 = editMe-e;
          h = getHVal(xVec);
          costMinus = getCost(h,y);
          approxDiff = (costPlus-costMinus)/(2.0*e);
          b2 = editMe;
          h = getHVal(xVec);
          double dJdb2 = h-y;
          b2diff = Math.pow(Math.abs(dJdb2-approxDiff), 2);
          
          System.out.println("b2 difference: " + b2diff);
          //L
          for (int i = 0; i < windowSize*wordSize; i++) {
            editMe = xVec.get(i, 0);
            xVec.set(i, 0, editMe+e);
            h = getHVal(xVec);
            costPlus = getCost(h,y);
            xVec.set(i,0, editMe-e);
            h = getHVal(xVec);
            costMinus = getCost(h,y);
            approxDiff = (costPlus-costMinus)/(2.0*e);
            xVec.set(i,0,editMe);
            z = W.mult(xVec).plus(b1);
            a = calcTanh(z);
            h = getHVal(xVec);
            dZ = derivTanh(z); 
            UTdZ = dZ.elementMult(U);
            delta2 = h - y;
            SimpleMatrix dJdL = W.transpose().mult(UTdZ).scale(delta2);
            Ldiff = Math.pow(Math.abs(dJdL.get(i)-approxDiff), 2);
          }
          System.out.println("L difference: " + Ldiff);
          

          if (Udiff > Math.pow(10, -7)) System.out.println("Gradient check FAILED with U difference of " + Udiff);
          if (Wdiff > Math.pow(10, -7)) System.out.println("Gradient check FAILED with W difference of " + Wdiff);
          if (b1diff > Math.pow(10, -7)) System.out.println("Gradient check FAILED with b1 difference of " + b1diff);
          if (b2diff > Math.pow(10, -7)) System.out.println("Gradient check FAILED with b2 difference of " + b2diff);
          if (Ldiff > Math.pow(10, -7)) System.out.println("Gradient check FAILED with L difference of " + Ldiff);
          else System.out.println("Gradient check PASSED");
        }
*/	
	public void test(List<Datum> testData){
          //System.out.println("Begin testing.");
          int predicted = 0;
          int actual = 0;
          int correct = 0;
          for (int i = 0; i < testData.size(); i++) {
            //System.out.println("Test window " + i + " of " + testData.size());
	    String[] window = getWindow(testData, i);
            SimpleMatrix xVec = new SimpleMatrix(windowSize*wordSize, 1);
            for (int j = 0; j < windowSize; j++) {
              int index = 0;
              if (FeatureFactory.wordToNum.containsKey(window[j])) index = FeatureFactory.wordToNum.get(window[j]);
              xVec.insertIntoThis(j*wordSize, 0, L.extractVector(true, index).transpose());
            }

            double feedforward = sigmoid(U.transpose().mult(calcTanh(W2.mult(calcTanh(W1.mult(xVec).plus(b1))).plus(b2))).get(0,0)+b3);
            //System.out.println(feedforward);
            int prediction = 0;
            if (feedforward >= 0.5) prediction = 1;
            int y = 1;
            if (testData.get(i).label.equals("O")) y = 0;
            if (prediction == 1) {
              predicted++;
              if (y == 1) correct++;
            } 
            if (y == 1) actual++;
            String predStr = "O";
            if (prediction == 1) predStr = "PERSON";
            //error analysis
            //if (y != prediction) System.out.println("Word: " + testData.get(i).word + " Prediction: " + predStr + " Actual: " + testData.get(i).label);
          }
          double precision = correct/((double)(predicted));
          double recall = correct/((double)(actual));
          double f1 = (2*precision*recall)/(precision+recall);
          System.out.println("Precision: " + precision);
          System.out.println("Recall: " + recall);
          System.out.println("F1: " + f1);
           
        }
	
}
