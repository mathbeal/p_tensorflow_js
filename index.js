const classifier = knnClassifier.create();
const webcamElement = document.getElementById('webcam');
const classes = ['Face', 'Book', 'Cellphone'];

let net;
let chart; // display classExampleCount

// Classifier functions
function get_class_example_infos(){
  console.log('get_class_exemple_infos');
  const counter = classifier.getClassExampleCount();
  var x_labels = classes;
  var y_values = [0, 0, 0];
  // update y_values if changed
  for (const [k, v] of Object.entries(counter)){
      y_values[k] = v;
  }
  // if not yet  classes example return default values
  return [x_labels, y_values];
}

// Chart functions
function update_chart(chart, x_labels, y_values, dataset_idx=0){
  console.log('update_chart');
  chart.data.labels = x_labels;
  chart.data.datasets[dataset_idx].data = y_values;
  chart.update();
}

function create_chart(){
  console.log('create_chart');
  const [x_labels, y_values] = get_class_example_infos();
  return new Chart(document.getElementById("bar-chart"), {
    type: 'bar',
    data: {
      labels: x_labels,
      datasets: [
        {
          label: "Number of samples",
          backgroundColor: ["#3e95cd", "#8e5ea2","#3cba9f"],
          data: y_values,
        }
      ]
    },
    options: {
      legend: { display: false },
      title: {
        display: true,
        text: 'Number of sample by class'
      },
      scales: {
        yAxes: [{
          ticks: {
            suggestedMin: 0,
            suggestedMax: 20,
          }
        }]
      }
    }
  });
}

chart = create_chart();

// Main application
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

      // Update chart
      const [x_labels, y_values] = get_class_example_infos();
      update_chart(chart, x_labels, y_values);

  };

  // When clicking a button, add an example for that class.
  document.getElementById('class-a').addEventListener('click', () => addExample(0));
  document.getElementById('class-b').addEventListener('click', () => addExample(1));
  document.getElementById('class-c').addEventListener('click', () => addExample(2));

  // Remove learned classes.
  document.getElementById('class-reset').addEventListener('click', function(){
	   classifier.clearAllClasses()
     const [x_labels, y_values] = get_class_example_infos();
     update_chart(chart, x_labels, y_values);
     $("#prediction").text("#");
     $("#probability").text("#");
     console.log('All learned classed are removed');
  });

  while (true) {

      // need at least one learned class.
      if (classifier.getNumClasses() > 0) {
      // Get the activation from mobilenet from the webcam.
      const activation = net.infer(webcamElement, 'conv_preds');
      // Get the most likely class and confidences from the classifier module.
	    const result = await classifier.predictClass(activation);

      $("#prediction").text(classes[result.classIndex]);
      // remove extra decimals
      // source: https://stackoverflow.com/questions/11832914/round-to-at-most-2-decimal-places-only-if-necessary
      $("#probability").text(Math.round(result.confidences[result.classIndex]*100)/100);

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

// navigation
$("#testing-nav").on('click', function(e){
  console.log('testing nav clicked');
  $("#training-content").hide();
  $("#training-nav").removeClass('active');
  $(this).addClass('active');
  $("#testing-content").show();
});

$("#training-nav").on('click', function(e){
  console.log('training nav clicked');
  $("#testing-content").hide();
  $("#testing-nav").removeClass('active');
  $(this).addClass('active');
  $("#training-content").show();
});

$(document).ready(function() {
    console.log('document rdy');
    $('#training-nav').click();

});

app();
