// https://storage.googleapis.com/tfjs-vis/mnist/dist/index.html
// https://towardsdatascience.com/common-loss-functions-in-machine-learning-46af0ffc4d23

const classifier = knnClassifier.create();
const webcamElement = document.getElementById('webcam');
const classes = ['Face', 'Book', 'Cellphone'];

let net;

async function app() {

  // Load the model.
  console.log('Loading mobilenet..');
  net = await mobilenet.load();
  console.log('Sucessfully loaded model');

  await setupWebcam();
  // Reads an image from the webcam and associates it with a specific class
  // index.

  const addExample = classId => {

      // Get the intermediate activation of MobileNet 'conv_preds' and pass that
      // to the KNN classifier.
      const activation = net.infer(webcamElement, 'conv_preds');

      // Pass the intermediate activation to the classifier.
      classifier.addExample(activation, classId);
  };

  // When clicking a button, add an example for that class.
  document.getElementById('class-a').addEventListener('click', () => addExample(0));
  document.getElementById('class-b').addEventListener('click', () => addExample(1));
  document.getElementById('class-c').addEventListener('click', () => addExample(2));

  // Saving and loading a model
  document.getElementById('class-savemodel').addEventListener('click', function(){
    console.log('save model');
  });

  document.getElementById('class-loadmodel').addEventListener('click', function(){
    console.log('load model');
  });

  // Remove learned classes.
  document.getElementById('class-reset').addEventListener('click', function(){
	   classifier.clearAllClasses()
     console.log('All classes cleared');
  });

  // tf.browser.fromPixels(document.getElementById('class1'));
  // source: https://github.com/tensorflow/tfjs-models/tree/master/knn-classifier
  document.getElementById('class-display').addEventListener('click', function(){
    const counter = classifier.getClassExampleCount();
    var x_labels = [];
    var y_values = [];
    for (const [k, v] of Object.entries(counter)){
        console.log(classes[k],v);
        x_labels.push(classes[k]);
        y_values.push(v);
    }
    new Chart(document.getElementById("bar-chart"), {
      type: 'bar',
      data: {
        labels: x_labels,
        datasets: [
          {
            label: "Number of samples",
            backgroundColor: ["#3e95cd", "#8e5ea2","#3cba9f","#e8c3b9","#c45850"],
            data: y_values,
          }
        ]
      },
      options: {
        legend: { display: false },
        title: {
          display: true,
          text: 'Number of sample by class'
        }
      }
    });
      var msg = '';
      msg = msg.concat(`Number of classed defined ${classifier.getNumClasses()}`);
      document.getElementById('display').innerText = msg;
    });

    // var errors
    // save trained set
    // load trained set
    // les images doivent avoir la meme taille ?

  while (true) {
      if (classifier.getNumClasses() > 0) {
      // Get the activation from mobilenet from the webcam.
      const activation = net.infer(webcamElement, 'conv_preds');
      // Get the most likely class and confidences from the classifier module.
	  const result = await classifier.predictClass(activation);

      document.getElementById('console').innerText = `
        last update: ${(new Date).toLocaleTimeString()}\n
        prediction: ${classes[result.classIndex]}\n
        probability: ${result.confidences[result.classIndex]}
      `;
    }

    await tf.nextFrame();
  }
}

async function setupWebcam() {
  return new Promise((resolve, reject) => {
    const navigatorAny = navigator;
    navigator.getUserMedia = navigator.getUserMedia ||
        navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
        navigatorAny.msGetUserMedia;
    if (navigator.getUserMedia) {
      navigator.getUserMedia({video: true},
        stream => {
          webcamElement.srcObject = stream;
          webcamElement.addEventListener('loadeddata',  () => resolve(), false);
        },
        error => reject());
    } else {
      reject();
    }
  });
}

app();
