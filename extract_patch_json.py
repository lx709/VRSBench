
import numpy as np
import os
import math
import cv2
import json
import argparse
# import requests
import openai
from openai import OpenAI
import base64
import pudb
import pickle
from termcolor import colored, cprint
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset', type=str, default='dota', help='dataset to use')
parser.add_argument('--data_split', type=str, default='train', help='dataset split')
parser.add_argument('--sel_patch', type=int, default=-1, help='select by grid index, only used for dior dataset')
parser.add_argument('--max_num_files', type=int, default=1000, help='Maximum number of files')
parser.add_argument('--shuffle', action='store_true', default=False, help='Whether to shuffle files')
parser.add_argument('--max_input_objects', type=int, default=10, help='Maximum input objects')
parser.add_argument('--use_obb', action='store_true', default=True, help='Whether to use OBB or HBB')
parser.add_argument('--save_vis', action='store_true', default=False, help='Flag to save visualization')
parser.add_argument('--start_idx', type=int, default=0, help='Start index of files')
parser.add_argument('--end_idx', type=int, default=99999, help='End index of files')
parser.add_argument('--output_root', type=str, default='dota_v10_val', help='output_root')
parser.add_argument('--exclude_roots', nargs='+', default=['dota_v10_val',], help='exclude_roots')

args = parser.parse_args()
cprint(args, 'green')

max_num_files = args.max_num_files
max_input_objects = args.max_input_objects
save_vis = args.save_vis

os.environ['OPENAI_API_KEY'] = 'YOUR_OPENAI_KEY'

client = OpenAI()

def box_close(bbox1, bbox2, dist_thres=0.1, use_obb=False):
    # calculate the distance between two boxes
    x1,y1,x2,y2,x3,y3,x4,y4 = bbox1
    x1_,y1_,x2_,y2_,x3_,y3_,x4_,y4_ = bbox2
    if use_obb==False:
        gap1 =  max(x1,x1_) - min(x3,x3_)
        gap2 =  max(y1,y1_) - min(y3,y3_)
        # print(gap1,gap2)
        if gap1<0 and gap2<0: # overlap
            return True
        elif gap1<0:
            return gap2<dist_thres
        elif gap2<0:
            return gap1<dist_thres
        else:
            return math.sqrt((gap1**2 + gap2**2)/2.0)<dist_thres
    else:
        gap1 =  max(x1,x1_) - min(x3,x3_)
        gap2 =  max(y1,y1_) - min(y3,y3_)
        # print(gap1,gap2)
        if gap1<0 and gap2<0: # overlap
            return True
        else:
            p1 = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
            p2 = np.array([[x1_,y1_],[x2_,y2_],[x3_,y3_],[x4_,y4_]])
            # calculate pairwise distance
            dist = np.zeros((4,4))
            for i in range(4):
                for j in range(4):
                    dist[i,j] = np.linalg.norm(p1[i,:]-p2[j,:])
            if np.min(dist)<dist_thres:
                return True
            else:
                return False

def get_size_from_points(bbox, image_shape, use_obb=True):
    bbox = bbox.copy()
    bbox[::2] = bbox[::2] * image_shape[1] # use absolute coordinates
    bbox[1::2] = bbox[1::2] * image_shape[0]
    if len(bbox)==4:
        xmin,ymin,xmax,ymax = bbox
        area = (xmax-xmin)*(ymax-ymin)
        return area
    x1,y1,x2,y2,x3,y3,x4,y4 = bbox
    if use_obb==False:
        xmin = min(x1,x2,x3,x4)
        xmax = max(x1,x2,x3,x4)
        ymin = min(y1,y2,y3,y4)
        ymax = max(y1,y2,y3,y4)
        area = (xmax-xmin)*(ymax-ymin)
        return area
    else:
        corner_points = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
        mask = np.zeros(image_shape, dtype=np.uint8)
        points = np.array(corner_points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 1)  # 255 to fill the polygon
        area = np.sum(mask) # / (image_shape[0] * image_shape[1])
        return area

# Function to read file content
def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Function to call the ChatGPT API
def call_chatgpt_api(instruction, input_text, base64_image=None, use_vision=True):
    if use_vision:
        model = "gpt-4-vision-preview"
    else:
        model="gpt-3.5-turbo-1106"
    # print('model: ', model)

    response = client.chat.completions.create(
        model=model,
        # response_format={ "type": "json_object"},
        messages=[
            {
                "role": "system", 
                "content": f"You are a helpful assistant designed to output JSON. {instruction}"
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text", 
                        "text": 'Input object information: ' + str(input_text),
                    },
                ]
            }
        ],
        max_tokens=1500,
    )
    
    return response.choices[0].message.content

# Function to write output to a JSON file
def write_to_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def main():
    data_root = './img_split/'
    ## read images
    if args.dataset=='dota':
        image_root = f'{data_root}/split_512_dota2_0/{args.data_split}/images/'
        cat_map = {
            "large-vehicle": "LV",
            "small-vehicle": "SV",
            "helicopter": "HC",
            "plane": "PL",
            "ship": "SH",
            "soccer-ball-field": "SBF",
            "basketball-court": "BC",
            "ground-track-field": "GTF",
            "baseball-diamond": "BD",
            "tennis-court": "TC",
            "roundabout": "RA",
            "storage-tank": "SC",
            "swimming-pool": "SP",
            "harbor": "HA",
            "container-crane": "CC",
            "airport": "AP",
            "helipad": "HP",
            "bridge": "bridge",
        }
    elif args.dataset=='dior':
        image_root = f'{data_root}/split_512_dior/trainval/images/'
        cat_map = {
            "airplane": "AP",
            "airport": "AR",
            "baseballfield": "BF",
            "basketballcourt": "BC",
            "bridge": "BR",
            "chimney": "CH",
            "expressway-service-area": "ESA",
            "expressway-toll-station": "ETS",
            "dam": "DM",
            "golffield": "GF",
            "groundtrackfield": "GTF",
            "harbor": "HB",
            "overpass": "OP",
            "ship": "SP",
            "stadium": "SD",
            "storagetank": "ST",
            "tenniscourt": "TC",
            "trainstation": "TS",
            "vehicle": "VE",
            "windmill": "WM",
        }
    else:
        raise ValueError('dataset not supported')

    if args.output_root is None:
        output_root = f'./{args.dataset}_labels_new/'
    else:
        output_root = args.output_root

    files = sorted(os.listdir(image_root))
    if args.dataset=='dior' and args.sel_patch in [0,1,2,3]:
        files = [file for file in files if file.endswith(f'_000{args.sel_patch}.png')]
    print('Number of files: ', len(files))

    if args.shuffle:
        np.random.shuffle(files)
    os.makedirs(output_root, exist_ok=True)

    # read instructions
    assert use_vision == True
    instruction = read_file("rs_instruction.txt")

    ## read labels
    if args.dataset=='dota':
        if args.use_obb:
            label_path = f'{data_root}/split_512_dota2_0/{args.data_split}/annfiles/'
        else:
            label_path = f'{data_root}/split_512_dota2_0_hbb/{args.data_split}/annfiles/'
    elif args.dataset=='dior':
        if args.use_obb:
            label_path = f'{data_root}/split_512_dior/trainval/annfiles/'
        else:
            label_path = f'{data_root}/split_512_dior/trainval/annfiles/'

    f = open(os.path.join(label_path, 'patch_annfile.pkl'), 'rb')
    anns = pickle.load(f)
    categories = anns['cls']
    anns = anns['content']
    f.close()

    # get average area for each category
    with open(f'bbox_size_stat_{args.dataset}.json', 'r') as f:
        area_cat = json.load(f)

    # start processing
    img_cnt = 0
    all_count_output = 0
    for fidx, file in enumerate(tqdm(files[args.start_idx:args.end_idx])):
        # unsafe images detected by openai
        if file in ['P1427_0151.png', 'P2285_0010.png', 'P1065_0022.png', 'P2612_0081.png', 'P2792_0002.png', 'P2693_0015.png', 'P0727_0003.png', 'P1426_0041.png', 'P1427_0001.png', 'P1427_0023.png', 'P1427_0048.png', 'P1431_0015.png', 'P1431_0022.png', 'P1431_0023.png', '00914_0003.png', '01592_0003.png', '03598_0001.png', '09800_0001.png', '00798_0001.png', '10524_0001.png', '05932_0001.png', '02236_0002.png', '02590_0002.png', '08742_0003.png', '04576_0002.png', '06209_0000.png', '07509_0000.png', '08966_0000.png', '03361_0002.png', '09244_0000.png', '03874_0002.png', '09320_0000.png', '06687_0003.png', '08060_0003.png', '03915_0003.png', '03747_0003.png', '07271_0002.png', '11537_0003.png']: 
            cprint(f'unsafe image {file}, skip this image!', 'red')
            continue

        if img_cnt>=max_num_files:
            break
        
        is_pass = False
        for exclude_root in args.exclude_roots:
            if os.path.exists(os.path.join(exclude_root, file.replace('png', 'json'))):
                cprint(f'file exists, skip this image, {file}', 'red')
                is_pass = True
                break
        if is_pass: continue

        img_path = os.path.join(image_root, file)
        idx = [i for i, ann in enumerate(anns) if ann['filename'] == file]
        item = anns[idx[0]]
        bboxes = item['ann']['bboxes'].astype('float32')
        if args.dataset == 'dior':
            xmin, ymin, xmax, ymax = bboxes[:,0], bboxes[:,1], bboxes[:,2], bboxes[:,3]
            bboxes = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
            bboxes = np.array(bboxes).T

        labels = item['ann']['labels'].astype('uint8')
        # diffs = item['ann']['diffs'].astype('bool')

        if args.dataset == 'dota':
            meta_file_path = os.path.join('dota_meta', file.split('_')[0]+'.txt')
            lines = open(meta_file_path).readlines()
            imagesource = lines[1].strip().split('imagesource:')[-1]
            gsd = lines[-1].strip().split('gsd:')[-1][:3]
            image_resolution = ''
            try:
                gsd = float(gsd)
                if gsd<1.0:
                    image_resolution = 'high'
                elif gsd<4.0:
                    image_resolution = 'medium'
                elif gsd>4:
                    image_resolution = 'low'
            except ValueError:
                pass
        else:
            imagesource = 'GoogleEarth'
            image_resolution = ''
    
        image = cv2.imread(img_path)
        image_shape = image.shape[:2]
        obj_id = 0
        json_input = {}
        refer_objects = []
        vis_all_objs = []

        if use_vision:
            # Getting the base64 string
            base64_image = encode_image(img_path)
        else:
            base64_image = None

        if len(bboxes)>max_input_objects: # ignore images with to many objects
            cprint(f'too many objects: {len(bboxes)}, skip this image!', 'yellow')
            continue
        
        per_img_valid = 0
        cat_cnt = {}
        for cat in categories:
            cat_cnt[cat] = np.sum(labels==categories.index(cat))

        bboxes[:,::2] = bboxes[:,::2]/image_shape[1]
        bboxes[:,1::2] = bboxes[:,1::2]/image_shape[0]
        
        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            
            cat = categories[label]
            x1,y1,x2,y2,x3,y3,x4,y4 = bbox
            # convert coordinates to float
            x1,y1,x2,y2,x3,y3,x4,y4 = map(float, [x1,y1,x2,y2,x3,y3,x4,y4])
            # horizontal bounding box
            xmin = max(min(x1,x2,x3,x4), 0)
            xmax = min(max(x1,x2,x3,x4), 1)
            ymin = max(min(y1,y2,y3,y4), 0)
            ymax = min(max(y1,y2,y3,y4), 1)

            # determine is the object the unique one in this category
            is_unique = np.sum(labels==label)==1
            
            # determine close objects from the same category
            same_cat_objs = list(np.where(labels==label)[0])
            same_cat_objs.remove(i)
            same_cat_bbox = bboxes[same_cat_objs]

            # determine object to exclude
            if 'small-vehicle' in cat:
                dist_thres = 0.2
            else:
                dist_thres = 0.15
            
            if any([box_close(bbox, bbox2, dist_thres) for bbox2 in same_cat_bbox]):
                is_close = True
            else:
                is_close = False

            is_exclude = cat_cnt[cat]>=10 or is_close

            # determine object color, not implemented yet

            # determine object size and relative size
            area = get_size_from_points(bbox, image_shape, args.use_obb)
            area_others = [get_size_from_points(bbox, image_shape, args.use_obb) for bbox in same_cat_bbox]
            
            obj_size = ''
            if cat not in ['small-vehicle', 'large-vehicle']: # not for vehicle
                if area>area_cat[cat][1]:
                    obj_size = 'large'
                elif area<area_cat[cat][0]:
                    obj_size = 'small'
                else:
                    obj_size = ''
            
            # import pudb; pudb.set_trace()
            obj_rel_size = ''
            if not is_unique:
                if all([area>ar*1.44 for ar in area_others]): # 1.2*1.2
                    if len(area_others)==1:
                        obj_rel_size = 'larger'
                    else:
                        obj_rel_size = 'largeest'
                elif all(area<ar*0.64 for ar in area_others): # 0.8*0.8
                    if len(area_others)==1:
                        obj_rel_size = 'smaller'
                    else:
                        obj_rel_size = 'smallest'
            
            # determine object absolute position
            if xmax<0.33 and ymax<0.33:
                obj_position = 'top-left'
            elif xmin>=0.33 and xmax<=0.66 and ymax<0.33:
                obj_position = 'top-middle'
            elif xmin>=0.66 and ymax<0.33:
                obj_position = 'top-right'
            elif xmax<0.33 and ymin>=0.33 and ymax<=0.66:
                obj_position = 'middle-left'
            elif xmin>=0.33 and xmax<=0.66 and ymin>=0.33 and ymax<=0.66:
                obj_position = 'center'
            elif xmin>=0.66 and ymin>=0.33 and ymax<=0.66:
                obj_position = 'middle-right'
            elif xmax<0.33 and ymin>=0.66:
                obj_position = 'bottom-left'
            elif xmin>=0.33 and xmax<=0.66 and ymin>=0.66:
                obj_position = 'bottom-middle'
            elif xmin>=0.66 and ymin>=0.66:
                obj_position = 'bottom-right'
            elif xmin>0.7:
                obj_position = 'right'
            elif xmax<0.3:
                obj_position = 'left'
            elif ymin>0.7:
                obj_position = 'bottom'
            elif ymax<0.3:
                obj_position = 'top'
            else:
                obj_position = ''

            # determine relative position
            obj_rel_position = ''
            # pudb.set_trace()
            if not is_unique:
                if np.min(bbox[1::2]) < np.min(same_cat_bbox[:,1::2]) - 30/image_shape[1]:
                    obj_rel_position = 'top-most'
                    is_exclude = False
                elif np.max(bbox[1::2]) > np.max(same_cat_bbox[:,1::2]) + 30/image_shape[1]:
                    obj_rel_position = 'bottom-most'
                    is_exclude = False
                elif np.min(bbox[::2]) < np.min(same_cat_bbox[:,::2]) - 30/image_shape[0]:
                    obj_rel_position = 'left-most'
                    is_exclude = False
                elif np.max(bbox[::2]) > np.max(same_cat_bbox[:,::2]) + 30/image_shape[0]:
                    obj_rel_position = 'right-most'
                    is_exclude = False
                else:
                    obj_rel_position = ''

            new_obj = {
                'obj_id': obj_id,
                'obj_cls': cat.lower(),
                'obj_corner': bbox.tolist(),
                'obj_coord': [round(xmin, 2), round(ymin, 2), round(xmax, 2), round(ymax, 2)],
                'is_unique': bool(is_unique), # avoid json decoding error
                'obj_position': obj_position,
                'obj_rel_position': obj_rel_position,
                # 'obj_color': nearest_color(mean_color[:3], common_colors),
                'obj_size': obj_size,
                'obj_rel_size': obj_rel_size,
                'flag': bool(is_exclude),
            }

            if i<=max_input_objects:
                if is_exclude==False:
                    per_img_valid += 1
                
                refer_objects.append(new_obj)
                obj_id += 1
            
            vis_all_objs.append(new_obj)
        
        if per_img_valid==0:
            cprint('all objects excluded, skip this image!', 'red')
            continue

        json_input = {'image source': imagesource, 'image resolution': image_resolution, 'refer_objects': refer_objects}
        in_file = os.path.join(output_root, file.split('.')[0] + '_input.json')
        # write_to_json(in_file, json_input)

        print('fidx, img_cnt, image, num boxes: ', fidx, img_cnt, file, len(bboxes))
        print('input json: ', json_input)

        is_valid = False
        for it in range(5):
            try:
                response = call_chatgpt_api(instruction, json_input, base64_image)
                # print('response json: ', response)
                response = response.split('```json')[-1]
                response = response.split('```')[0]
                response = response.replace('True', 'true')
                response = response.replace('False', 'false')
                response_json = json.loads(response)
                exclude_phrases = ['flag', 'not provide', 'not specified', 'unknown', 'referred', 'referring',\
                        'nose', 'vertical stabilizer', ' tail', 'tail ', 'facing', 'pointing',\
                        'first-mentioned', 'aforementioned', 'previously mentioned', 'motion', 'day', 'night', \
                        # 'distinctive', 'distinguishable',
                        ]
                
                caption = response_json['caption']
                referring_sentence = "\n".join([res["referring_sentence"] for res in response_json['objects']])
                qa_pairs = "\n".join([res["question"] for res in response_json['qa_pairs']])
                all_checks = caption + referring_sentence + qa_pairs
                if not any([ep in all_checks for ep in exclude_phrases]):
                    is_valid = True
                    break
                else:
                    # print the excluded phrases
                    for ep in exclude_phrases:
                        if ep in all_checks:
                            cprint(f'exclude_phrases {ep} in response json, try again!', 'yellow')
                    is_valid = False
            # except openai.BadRequestError:
            #     cprint('openai.BadRequestError, try again!', 'yellow')
            except KeyError:
                cprint('KeyError, try again!', 'yellow')
                pass
            except json.decoder.JSONDecodeError:
                cprint('JSONDecodeError, try again!', 'yellow')
                pass
            finally:
                # exclude phrases
                cap_sents = response_json['caption'].split('. ')
                cap_sents = [res for res in cap_sents if not any([ep in res for ep in exclude_phrases])]
                cap_sents = ". ".join(cap_sents)
                if not cap_sents.endswith('.'):
                    cap_sents += '.'
                response_json['caption'] = cap_sents
                
                response_json['objects'] = [res for res in response_json['objects'] if not any([ep in res["referring_sentence"] for ep in exclude_phrases])]
                response_json['qa_pairs'] = [res for res in response_json['qa_pairs'] if not any([ep in res["question"] for ep in exclude_phrases])]
        
        if not is_valid:
            cprint('still have exclude_phrases, skip this image', 'red')
            continue
        
        # import pudb; pudb.set_trace()
        response_json['image'] = file
        output_file = os.path.join(output_root, file.replace('png', 'json'))
        # retrain original attributes if not returned (GPT-4 can lose information)
        for obj in response_json['objects']:
            obj_id = obj['obj_id']
            obj.update(json_input['refer_objects'][obj_id])
        print('response json: ', response_json)
        write_to_json(output_file, response_json)

        img_cnt += 1

        # save visualized image
        if save_vis:
            count_output = 0
            for obj in vis_all_objs: # all objects, refer_objects can ignore some objects
                obj_id = obj['obj_id']
                obj_cls = obj['obj_cls']
                obj_cls = cat_map[obj_cls]
                color = (0, 255, 0) 
                thickness = 1
                if args.use_obb:
                    bbox = np.array(obj['obj_corner'])
                    bbox = bbox.copy()
                    bbox[::2] = bbox[::2] * image_shape[1] # convert to absolute coordinates
                    bbox[1::2] = bbox[1::2] * image_shape[0]
                    bbox = bbox.astype('int32')
                    x1, y1, x2, y2, x3, y3, x4, y4 = bbox
                    # draw lines
                    image = cv2.line(image, (x1, y1), (x2, y2), color, thickness)
                    image = cv2.line(image, (x2, y2), (x3, y3), color, thickness)
                    image = cv2.line(image, (x3, y3), (x4, y4), color, thickness)
                    image = cv2.line(image, (x4, y4), (x1, y1), color, thickness)
                else:
                    obj_coord = obj['obj_coord']
                    x1, y1, x2, y2 = obj_coord
                    x1,y1,x2,y2 = map(int, [x1*image_shape[1],y1*image_shape[0],x2*image_shape[1],y2*image_shape[0]])
                    start_point = (x1, y1) 
                    end_point = (x2, y2) 
                    image = cv2.rectangle(image, start_point, end_point, color, thickness)
                # show object id and class on the top left corner
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.4
                org = (x1, y1-10)
                image = cv2.putText(image, f'{obj_id}: {obj_cls}', org, font, fontScale, color, thickness, cv2.LINE_AA)

            for obj in response_json['objects']: # referring objects
                obj_id = obj['obj_id']
                obj_cls = obj['obj_cls']
                obj_cls = cat_map[obj_cls]
                color = (0, 0, 255) 
                thickness = 1
                if args.use_obb:
                    bbox = np.array(obj['obj_corner'])
                    bbox = bbox.copy()
                    bbox[::2] = bbox[::2] * image_shape[1] # convert to absolute coordinates
                    bbox[1::2] = bbox[1::2] * image_shape[0]
                    bbox = bbox.astype('int32')
                    x1, y1, x2, y2, x3, y3, x4, y4 = bbox
                    # draw lines
                    image = cv2.line(image, (x1, y1), (x2, y2), color, thickness)
                    image = cv2.line(image, (x2, y2), (x3, y3), color, thickness)
                    image = cv2.line(image, (x3, y3), (x4, y4), color, thickness)
                    image = cv2.line(image, (x4, y4), (x1, y1), color, thickness)
                else:
                    obj_coord = obj['obj_coord']
                    x1, y1, x2, y2 = obj_coord
                    x1,y1,x2,y2 = map(int, [x1*image_shape[1],y1*image_shape[0],x2*image_shape[1],y2*image_shape[0]])
                    start_point = (x1, y1) 
                    end_point = (x2, y2) 
                    image = cv2.rectangle(image, start_point, end_point, color, thickness)
                # show object id and class on the top left corner
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.4
                org = (x1, y1-10)
                image = cv2.putText(image, f'{obj_id}: {obj_cls}', org, font, fontScale, color, thickness, cv2.LINE_AA)

                count_output += 1

            # Displaying the image  
            cv2.imwrite(os.path.join(output_root, file), image)  
            print('Per image number of referred objects: ', count_output)
            all_count_output += count_output

    print('All images number of referred objects: ', all_count_output)
        

if __name__ == "__main__":
    main()
