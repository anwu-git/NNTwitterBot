import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#load the pretrained neural network model
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnext50_32x4d', pretrained=True)
model.eval()

#PIL to load image from a file
from PIL import Image 
from torchvision import transforms

#All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape 
#(3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then 
#normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

preprocess_default = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), #transforms a PIL image or numpy.ndarray to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

preprocess_random_perspective = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

preprocess_flip_horizontal = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

preprocess_tilt = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomRotation(degrees=(-15, 15)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def resnext_run(filename, preprocess=preprocess_default):
    # sample execution (requires torchvision)
    input_image = Image.open(filename).convert('RGB')
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    #if torch.cuda.is_available():
    #    input_batch = input_batch.to('cuda')
    #    model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    return probabilities

def resnext_classify(filename, printout=False):

    probabilities = resnext_run(filename)

    print(probabilities.size())

    #with function: It’s handy when you have two related operations which you’d like to execute as a pair 
    # and you need the execution of both of them to be guaranteed no matter how the nested code in between them might exit. 
    with open("testimg/imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    if printout:
        for i in range(top5_prob.size(0)):
            print(categories[top5_catid[i]], top5_prob[i].item())

    topfive_cats  = []
    topfive_probs = []
    topfive_string = []
    
    for i in range(top5_prob.size(0)):
        topfive_cats.append(categories[top5_catid[i]])
        topfive_probs.append(top5_prob[i].item())
        topfive_string.append("#" + str(i+1) + ": " + categories[top5_catid[i]] + " (Confidence: " + str(round(top5_prob[i].item()*100, 2)) + "%)\n")
        
    #\n line break isn't being used properly in our string so we translate our tuple of strings and 
    # ints into just a list of strings and use str.join() to join them together into one big string
    # via https://stackoverflow.com/a/21068541
    topfive_string_mapped = map(str, topfive_string)
    topfive_string_imglegend = ''.join(topfive_string_mapped)
    
    #opens a window and shows the image
    img = mpimg.imread(filename)
    plt.imshow(img)
    plt.text(1,15,topfive_string_imglegend,fontdict={'fontsize': 6, 'fontweight': 'medium'})
    plt.show()
    
    return [topfive_cats, topfive_probs, top5_catid]

def test():
    print(resnext_classify("testimg/test.jpg", printout=True))
    
if __name__ == '__main__':
    test()