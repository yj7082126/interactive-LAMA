// SETTING ALL VARIABLES

var isMouseDown = false;
var canvas = document.createElement("canvas");
var body = document.getElementById("container_col");
var ctx = canvas.getContext("2d");

var linesArray = [];
var currentSize = 20;
var currentColor = "#ff00ff";

var myBoard = new DrawingBoard.Board("zbeubeu", {
  controls: [
    "Color",
    { Size: { type: "dropdown" } },
    "DrawingMode",
    "Navigation",
    "Download"
  ], 
  background: "/img/tigerbro_clean_v2/001-000-2a5ce1afb166.jpg",
  color: currentColor,
  size : 20,
  eraserColor : "transparent",
  enlargeYourContainer: true,
  stretchImg: true
});
myBoard.clearWebStorage();

// BUTTON EVENT HANDLERS
document.getElementById("uploadimage").addEventListener( "click", function() { upload("canvas"); }, false);
document.getElementById("left-button").addEventListener( "click", function() { change("left"); }, false);
document.getElementById("right-button").addEventListener( "click", function() { change("right"); }, false);

function apply(x, width, height){
  console.log(width);
  console.log(height);

  var width2  = parseInt(width, 10) + parseInt("2px", 10) + "px";
  var height2 = parseInt(height, 10) + parseInt("35px", 10) + "px";
  $(".drawing-board").css("width",  width2).css("height", height2);
  $(".drawing-board-canvas-wrapper").css("width",  width).css("height", height);
  $(".drawing-board-canvas").css("width",  width).css("height", height);      
  myBoard.canvas.width = width;
  myBoard.canvas.height = height;
  myBoard.setImg(x);

  myBoard.ctx.lineWidth = currentSize;
  myBoard.setColor(currentColor);
}

function init_img(dir){
  console.log(dir);
  let img = document.getElementById("image");
  let x = "/img/" + dir;
  img.onload = function() { apply(x, img.width, img.height); }
  img.src = x;
}

// Move to next img
function change(type) {
  fetch("/change", {method: "POST", body: type})
    .then(response => response.json())
    .then(r => {
      let img = document.getElementById("image");
      let x = "/" + r.location;
      // var img = new Image();
      img.onload = function() {   
        console.log(this.width);
        console.log(this.height);
        
        var width2  = parseInt(this.width, 10) + parseInt("2px", 10) + "px";
        var height2 = parseInt(this.height, 10) + parseInt("35px", 10) + "px";
        $(".drawing-board").css("width",  width2).css("height", height2);
        $(".drawing-board-canvas-wrapper").css("width",  this.width).css("height", this.height);
        $(".drawing-board-canvas").css("width",  this.width).css("height", this.height);      
        myBoard.canvas.width = this.width;
        myBoard.canvas.height = this.height;
        myBoard.setImg(x);

        myBoard.ctx.lineWidth = currentSize;
        myBoard.setColor(currentColor); 
      }
      img.src = x;
    });
}

// UPLOAD CANVAS
function upload(canvas) {
  // TODO change
  var dataURL = myBoard.getImg();

  let img = document.getElementById("image");
  img.src = "/img/loading.gif"; // TODO fix this to use a better gif

  fetch("/upload", { method: "POST", body: dataURL})
    .then(response => response.json())
    .then(r => {
      let x = "/" + r.location;
      img.src = x;
    });
}