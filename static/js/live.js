var imgData
(function() {
        var width = 650; // We will scale the photo width to this
        var height = 0; // This will be computed based on the input stream
        var streaming = false;

        var video = null;
        var canvas = null;
        var photo = null;
        var startbutton = null;

        function startup() {
            video = document.getElementById('video');
            canvas = document.createElement('canvas');
            photo = document.getElementById('unknown_image');
            startbutton = document.getElementById('capture-btn');

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

            imgData = canvas.toDataURL('image/jpg');
            photo.setAttribute('src', imgData);
        }

        function takepicture() {
            var context = canvas.getContext('2d');
            if (width && height) {
                canvas.width = width;
                canvas.height = height;
                context.drawImage(video, 0, 0, width, height);

                imgData = canvas.toDataURL('image/jpg');
                photo.setAttribute('src', imgData);
            } else {
                clearphoto();
            }
        }

        window.addEventListener('load', startup, false);
    })();

$("#info-submit").click(function(e) {
                $.ajax({
                        url: '/recognizefromcamera',
                        type: 'post',
                        data: JSON.stringify({imageBase64: imgData}),
                        contentType: "application/json; charset=utf-8",
                        dataType: 'json',
                        success: (function(data){
                        $("#dist").text(data['dist']);
                        $("#label").text(data['label']);
//                        console.log(data['dist']);
//                        console.log(data['label']);
                        })
})
});
