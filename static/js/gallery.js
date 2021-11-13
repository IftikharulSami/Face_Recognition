var imgdata = null;
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
imgdata = canvas.toDataURL('image/jpg');
resizeImg.setAttribute('src', imgdata);
document.getElementById('base64').innerHTML = imgdata;
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
                        url: '/addimagefromgallery',
                        type: 'post',
                        data: JSON.stringify({label: name, imageBase64: imgdata}),
                        contentType: "application/json; charset=utf-8",
                        dataType: 'json',
                        }).done(function(data){
                        $("#response").text(data['resp']);
                        });
    });