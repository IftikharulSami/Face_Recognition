var Imgdata;
     (function() {
        var width = 680; // We will scale the photo width to this
        var height = 0; // This will be computed based on the input stream
        var streaming = false;

        var video = null;
        var canvas = null;
        var canvas2 = null;
        var photo = null;
        var resizePhoto = null;
        var startbutton = null;

        function startup() {
            video = document.getElementById('video');
            canvas = document.createElement('canvas');
            photo = document.getElementById('photo');
            startbutton = document.getElementById('capture');

            navigator.mediaDevices.getUserMedia({
                    video: true,
                    audio: false
                })
                .then(function(stream) {
                    video.srcObject = stream;
                    video.play();
                })
                .catch(function(err) {
                    console.log("An error occurred: " + err);
                });

            video.addEventListener('canplay', function(ev) {
                if (!streaming) {
                    height = video.videoHeight / (video.videoWidth / width);

                    if (isNaN(height)) {
                        height = width / (4 / 3);
                    }

                    video.setAttribute('width', width);
                    video.setAttribute('height', height);
                    canvas.setAttribute('width', width);
                    canvas.setAttribute('height', height);
                    canvas2.setAttribute('width', width);
                    canvas2.setAttribute('height', height);
                    streaming = true;
                }
            }, false);

            startbutton.addEventListener('click', function(ev) {
                takepicture();
                ev.preventDefault();
            }, false);

            clearphoto();
        }

        function clearphoto() {
            var context = canvas.getContext('2d');
            context.fillStyle = "#AAA";
            context.fillRect(0, 0, canvas.width, canvas.height);

            var data = canvas.toDataURL('image/jpg');
            photo.setAttribute('src', data);
        }

        function takepicture() {
            var context = canvas.getContext('2d');
            if (width && height) {
                canvas.width = width;
                canvas.height = height;
                context.drawImage(video, 0, 0, width, height);

                var data = canvas.toDataURL('image/jpg');
                photo.setAttribute('src', data);
            } else {
                clearphoto();
            }
        }

        window.addEventListener('load', startup, false);
    })();

function resizeImage(){
var width = parseInt(document.getElementById('txt-width').value);
var height = parseInt(document.getElementById('txt-height').value);
var resizeImg = document.getElementById('resizedPhoto');
var img = document.getElementById('photo').src;
var canvas = document.createElement('canvas');
//if (img){
//document.getElementById('msg').innerHTML = img;
//}
var context = canvas.getContext('2d');
make_base();
function make_base(){
base_img = new Image();
base_img.src = img;
if (width && height) {
canvas.width = width;
canvas.height = height;
context.drawImage(base_img, 0, 0, width, height);
Imgdata = canvas.toDataURL('image/jpg');
resizeImg.setAttribute('src', Imgdata);
}}
document.getElementById("resize").style["width"] = width+'px';
document.getElementById("resize").style["height"] = height+'px';
}

document.getElementById('enable-txt').onchange = function() {
    document.getElementById('txt-width').disabled = !this.checked;
    document.getElementById('txt-height').disabled = !this.checked;
};

$("#info-submit").click(function(e) {
    var name = $("#name").val();
                $.ajax({
                        url: '/addimagefromcamera',
                        type: 'post',
                        data: JSON.stringify({imageBase64: Imgdata, label: name}),
                        contentType: "application/json; charset=utf-8",
                        dataType: 'json',
                        }).done(function(data){
                        $("#response").text(data['resp']);
                        });
    });