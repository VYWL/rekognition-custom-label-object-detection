import boto3
import io
import os
from PIL import Image, ImageDraw, ExifTags, ImageColor
from dotenv import load_dotenv

load_dotenv(verbose=True)

def show_objects(photo, bucket, PrjVerArn):

    client=boto3.client('rekognition')

    # Load image from S3 bucket
    s3_connection = boto3.resource('s3')
    s3_object = s3_connection.Object(bucket,photo)
    s3_response = s3_object.get()

    stream = io.BytesIO(s3_response['Body'].read())
    image=Image.open(stream)
    
    #Call DetectCustomLabels 
    response = client.detect_custom_labels(
        ProjectVersionArn=PrjVerArn, 
        Image={
            'S3Object': {
                'Bucket': bucket, 
                'Name': photo,
            }
        },
        MaxResults=1,
        MinConfidence=0
    )

    imgWidth, imgHeight = image.size  
    draw = ImageDraw.Draw(image)  
                    

    # calculate and display bounding boxes for each detected face       
    print('Detected Objects for ' + photo)  

    for objDetail in response['CustomLabels']:
        print('The confidence of detected object is ' + str(objDetail["Confidence"]))
        
        box = objDetail["Geometry"]['BoundingBox']
        left = imgWidth * box['Left']
        top = imgHeight * box['Top']
        width = imgWidth * box['Width']
        height = imgHeight * box['Height']
                

        print('Left: ' + '{0:.0f}'.format(left))
        print('Top: ' + '{0:.0f}'.format(top))
        print('Face Width: ' + "{0:.0f}".format(width))
        print('Face Height: ' + "{0:.0f}".format(height))

        points = (
            (left,top),
            (left + width, top),
            (left + width, top + height),
            (left , top + height),
            (left, top)

        )
        draw.line(points, fill='#00d400', width=10)

        # Alternatively can draw rectangle. However you can't set line width.
        #draw.rectangle([left,top, left + width, top + height], outline='#00d400') 
    image.save(f'%s.png'%{photo}, 'png')

    return len(response['CustomLabels'])

def main():
    bucket = os.getenv('BUCKET_NAME')
    photo = os.getenv('PHOTO_NAME')
    prjArn = os.getenv('PRJVERARN')
    
    objects_count = show_objects(photo,bucket,prjArn)
    print("Object detected: " + str(objects_count))


if __name__ == "__main__":
    main()