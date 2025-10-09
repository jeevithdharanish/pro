
import { Copy } from 'lucide-react';
import  { useState } from 'react';

export default function CodeGallery() {
  const codes = [
  {
    title: "1️ Image Histogram",
    filename: "image_histogram.m",
    code: `clc; 
clear; 
img = imread("D:\dip.jpg") 
if ndims(img) == 3 then 
    img1 = rgb2gray(img); 
else 
    img1 = img; 
end 
hist = histc(img1(:), 0:255);  
subplot(1,3,1); 
imshow(img); 
title("Original Image"); 
subplot(1,3,2); 
imshow(img1); 
title("Gray scale Image"); 
subplot(1,3,3); 
bar(hist, 'b'); 
title("Image Histogram"); 
xlabel("Pixel Value"); 
ylabel("Frequency"); `
  },
  {
    title: "2️ Histogram Equalization",
    filename: "histogram_equalization.m",
    code: `clc; 
clear; 
img = imread("D:\dip.jpg");   
if ndims(img) == 3 then 
    img = rgb2gray(img); 
end 
img = im2double(img); 
hist_original = histc(img(:), 0:0.01:1);   
cdf = cumsum(hist_original); 
cdf_normalized = (cdf - min(cdf)) / (max(cdf) - min(cdf)); 
img_eq = interp1(linspace(0, 1, length(cdf)), cdf_normalized, img, "linear", 0); 
hist_equalized = histc(img_eq(:), 0:0.01:1);  
subplot(2, 3, 1); 
imshow(img); 
title("Original Image"); 
subplot(2, 3, 2); 
imshow(img_eq); 
title("Equalized Image"); 
subplot(2, 3, 3); 
bar(hist_original, 'b'); 
title("Histogram of Original Image"); 
xlabel("Pixel Value"); 
ylabel("Frequency"); 
subplot(2, 3, 4); 
bar(hist_equalized, 'r'); 
title("Histogram of Equalized Image"); 
xlabel("Pixel Value"); 
ylabel("Frequency"); `
  },
  {
    title: "3️ Image Enhancement using Intensity Mapping",
    filename: "intensity_mapping.m",
    code: `img=imread("D:\dip.jpg"); 
maping = imadjust(img, [0.3 0.7], [0 1]); 
subplot(1,2,1); 
imshow(img); 
title("Original Image"); 
subplot(1,2,2); 
imshow(maping); 
title("Enhanced Image"); `
  },
  {
    title: "4️ Image Arithmetic Operations",
    filename: "image_arithmetic.m",
    code: `clc; 
clear; 
close; 
i = imread("D:\dip.jpg"); 
j = imread("D:\dip.jpg"); 
k = imadd(i, j); 
subplot(3,2,1); 
imshow(i); 
title("Original Image 1"); 
subplot(3,2,2); 
imshow(j); 
title("Original Image 2"); 
subplot(3,2,3); 
imshow(k); 
title("Addition"); 
d = imabsdiff(i, j); 
subplot(3,2,4); 
imshow(d); 
title("Difference"); 
m = immultiply(i, j); 
subplot(3,2,5); 
imshow(m); 
title("Multiply"); 
v = imdivide(i, j); 
subplot(3,2,6); 
imshow(v); 
title("divide"); `
  },
  {
    title: "5️ High Pass and Low Pass Filters",
    filename: "filters_high_low.m",
    code: `clc; 
clear; 
img = imread("D:\dp.jpg"); 
if size(img,"c")==3 then 
    img = rgb2gray(img); 
end 
a = im2double(img); 
subplot(1,3,1); 
imshow(img); 
title("Original Image"); 
[m,n] = size(a); 
// High-pass 
w = [-1 -1 -1; -1 8 -1; -1 -1 -1]; 
b = zeros(m,n); 
for i=2:m-1 
    for j=2:n-1 
        region = a(i-1:i+1,j-1:j+1); 
        b(i,j) = sum(sum(w.*region)); 
    end 
end 
c = im2uint8(b); 
subplot(1,3,2); 
imshow(c); 
title("High pass filter"); 
// Low-pass 
u = ones(3,3); 
b = zeros(m,n); 
for i=2:m-1 
    for j=2:n-1 
        region = a(i-1:i+1,j-1:j+1); 
        b(i,j) = sum(sum(u.*region))/9; 
    end 
end 
d = im2uint8(b); 
subplot(1,3,3); 
imshow(d); 
title("Low pass filter"); `
  },
  {
    title: "6️ Geometric Transformations",
    filename: "geometric_transformations.m",
    code: `clc; 
clear; 
close; 
img = imread("D:\dp.jpg"); 
if size(img, "c") == 3 then 
    img = rgb2gray(img); 
end 
subplot(3,3,1); 
imshow(img); 
title("Original Image"); 
scaled_img = imresize(img, [round(size(img,1)*1.5), round(size(img,2)*1.2)]); 
subplot(3,3,2); 
imshow(scaled_img); 
title("Scaled Image"); 
rotated_img = imrotate(img, 45);    
subplot(3,3,3); 
imshow(rotated_img); 
title("Rotated Image"); `
  },
  {
    title: "7️ Image Transformations (Negative, Brightness, Contrast)",
    filename: "image_transformations.m",
    code: `clc; 
clear; 
close; 
I = imread("D:\dp.jpg"); 
I = im2double(I); // convert to range [0,1] for processing 
// ----- Negative Transformation ----- 
Neg = 1 - I; 
// ----- Brightness Increase ----- 
Bright = I + 0.3; // add constant 
Bright(Bright > 1) = 1; // clip to max value 
// ----- Contrast Stretching ----- 
minI = min(I(:)); 
maxI = max(I(:)); 
Contrast = (I - minI) ./ (maxI - minI); // stretch to [0,1] 
// ----- Display Results ----- 
subplot(2,2,1); 
imshow(I); 
title("Original Image"); 
subplot(2,2,2); 
imshow(Neg); 
title("Negative Image"); 
subplot(2,2,3); 
imshow(Bright); 
title("Brightness Increased"); 
subplot(2,2,4); 
imshow(Contrast); 
title("Contrast Stretched"); `
  },
  {
    title: "8️ Morphological Operations",
    filename: "morphological_operations.m",
    code: `clc; 
clear; 
close; 
sbin=imread("D:\dip.jpg")  
subplot(1,3,1)  
imshow(sbin);  
title('Orignial Image')  
se=imcreatese('ellipse',15,15);  
sd=imdilate (sbin,se);  
subplot(1,3,2)  
imshow(sd);  
title('Dilated image')  
se=imerode (sbin,se);  
subplot(1,3,3)  
imshow(se);  
title('Eroded image') `
  },
  {
    title: "9️ RGB Channel Separation",
    filename: "rgb_channel_separation.m",
    code: `clc; 
clear; 
close; 
RGB=imread("D:\dip.jpg")  
// Show original image 
subplot(2,2,1); 
imshow(RGB); 
title("Original"); 
// Extract R, G, B channels 
R = RGB(:,:,1); 
G = RGB(:,:,2); 
B = RGB(:,:,3); 
// Show Red channel 
subplot(2,2,2); 
imshow(R); 
title("Red&quot"); 
// Show Green channel 
subplot(2,2,3); 
imshow(G); 
title("Green&quot"); 
// Show Blue channel 
subplot(2,2,4); 
imshow(B); 
title("Blue");`
  },
  {
    title: " Noise Addition and Edge Detection using LOG",
    filename: "noise_edge_detection.m",
    code: `clc;  
clear; 
j=imread("D:\dip.jpg");  
j1=rgb2gray(j);  
n=25*rand(size(j1,1),size(j1,2));  
subplot(3,2,1);  
imshow(j);  
title("Original Image");  
subplot(3,2,2);  
imshow(j1);  
title("Gray scale Image");  
subplot(3,2,3);  
imshow(n);  
title("Generated noise");  
j2=n+j1;  
subplot(3,2,4);  
imshow(j2);  
title("Noise+Image")  
Lap=[0 -1 0; -1 4 -1; 0 -1 0];  
j2=double(j2)  
j3=conv2(j2, Lap, 'same');  
subplot(3,2,5);  
imshow(j3);  
title("Detected edge after LOG");`
  }
];


  const [copiedIndex, setCopiedIndex] = useState(null);

  const handleCopy = async (index, text) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedIndex(index);
      setTimeout(() => setCopiedIndex(null), 1500);
    } catch (err) {
      console.error("Copy failed:", err);
    }
  };

  return (
    <div className="page-container">
      <div className="gallery-grid">
        {codes.map((item, index) => (
          <div key={index} className="code-card">
            <div className="card-header">
              <div>
                <h2 className="card-title">{item.title}</h2>
                <p className="card-filename">{item.filename}</p>
              </div>
              <button
                onClick={() => handleCopy(index, item.code)}
                className="copy-button"
              >
                <Copy size={16} />
                {copiedIndex === index ? "Copied!" : "Copy"}
              </button>
            </div>
            <pre className="code-block">
              <code>{item.code}</code>
            </pre>
          </div>
        ))}
      </div>
    </div>
  );
}