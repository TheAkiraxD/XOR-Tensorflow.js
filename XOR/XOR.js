// XOR Table
// _____________
//| 1 | 2 | XOR |
//| T | T |  F  |
//| T | F |  T  |
//| F | T |  T  |
//| F | F |  F  |
// ͞ ͞ ͞ ͞ ͞ ͞ ͞ ͞ ͞ ͞ ͞ ͞ ͞          

const train_xs = tf.tensor2d([[0,0], [1,0], [0,1], [1,1]]);
const train_ys = tf.tensor2d([[0], [1], [1], [0]]);

var model;
var resolution = 30;
var cols;
var rows;
var inputs = [];
var xs;

function setup() {
    createCanvas(600, 600);
    //frameRate(30);
    cols = width / resolution;
    rows = height / resolution;

    // Creating the inputs array
    for(var i = 0; i < cols; i++){
        for(var j = 0; j < rows; j++){
            var x1 = i / cols;
            var x2 = j / rows;
            inputs.push([x1,x2]);
        }
    }
    xs = tf.tensor2d(inputs);


    // Model which contains the whole Neural Network
    model = tf.sequential();

    // Hidden layer
    var hidden = tf.layers.dense({
        inputShape: [2],
        units: 2,
        activation: 'sigmoid'
    });
    model.add(hidden);

    // Output layer
    var output = tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    });
    model.add(output);

    // Configuring the Neural Network
    const optimizer = tf.train.sgd(0.5);

    model.compile({
        optimizer: optimizer,
        loss: 'meanSquaredError'
    });
}

function trainModel(){
    return model.fit(train_xs,train_ys, {
        shuffle: true,
        epochs: 10
    });
}

function draw() {
    background(0);
    tf.tidy(() => {
        trainModel();

        // Predictions
        var ys = model.predict(xs).dataSync();

        // Drawing
        var index = 0;
        for(var i = 0; i < cols; i++){
            for(var j = 0; j < rows; j++){
                var br = ys[index] * 255;
                fill(br);
                rect(i * resolution, j * resolution, resolution, resolution);
                fill('rgba(255,0,0, 0.5)');
                textAlign(CENTER, CENTER);
                text(nf(ys[index],1,2), i * resolution + resolution/2, j * resolution + resolution/2);
                index++;
            }
        }

    });


}

