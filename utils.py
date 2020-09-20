# encoding=utf-8
import sys
sys.path.append('/workspace/external-libraries/')

import jieba
import os
import json
import numpy as np
import h5py
from scipy.misc import imread,imresize
from tqdm import tqdm
import torch
from random import seed, choice, sample
from PIL import Image

def bulid_data(data_root,filename): 
    """
    json_file ==> entity_dict(image_name,sent,sents_token)
    """
    entity = {}
    sents = []
    sents_tokenize = []
    with open(os.path.join(data_root,filename),'r') as f:
        data = json.load(f)
    for data_part in data['response']['annotations']:
        if len(data_part['attributes']) > 5:
            seg_list = jieba.cut(data_part['attributes'].strip().replace(u'。',''),cut_all = False)
            sents_tokenize.append(list(seg_list))
            sents.append(data_part['attributes'])
    img_name = annotation.split('_')[0]
    if sents != []:
        entity['image_name'] = img_name
        entity['sents'] = sents
        entity['sents_token'] = sents_tokenize
        return entity
    else:
        return {}
    
def bulid_vocab(imgs):
    """
    all entity ==> vocab
    """
    param = {}
    counts = {}
    # bulid_vocab
    for img in imgs:
        #print(img['sents_token'])
        for sent in img['sents_token']:
            for w in sent:
                counts[w] = counts.get(w,0) + 1
    cw = sorted([(count,w) for w,count in counts.items()],reverse=True)            
    #print(cw)
    bad_word = [w for w,n in counts.items() if n <= 1]
    #print(len(bad_word),len(counts))
    vocab = [w for w,n in counts.items() if n > 1]
    print('number of vocab is {}'.format(len(vocab)))
    bad_count = sum(counts[w] for w in bad_word)
    sent_length = {}
    for img in imgs:
        for sent in img['sents_token']:
            nw = len(sent)
            sent_length[nw] = sent_length.get(nw,0) +1
    max_len = max(sent_length.keys())
    param['max_length'] = max_len
    sum_len = sum(sent_length.values())
       
    vocab.append(u'<unk>')
    vocab.append(u'<start>')
    vocab.append(u'<end>')
    vocab.insert(0,u'<pad>')
    
    #print(vocab)
    for img in imgs:
        img['final_captions'] = []
        for sent in img['sents_token']:
            caption = [w if counts.get(w,0) > 1 else u'UNK' for w in sent]
            caption.append(u'<end>')
            caption.insert(0,u'<start>')
            img['final_captions'].append(caption)
    return vocab,param

def create_input_files(imgs,split, params, word_map,image_root):
    """
    generate files
    """
    output_folder = './process_data_2/'
    max_len = param['max_length'] + 2 
    image_name_data = os.listdir(image_root)
    with h5py.File(os.path.join(output_folder,'{}_IMAGE.hdf5'.format(split)),'a') as h:
        h.attrs['captions_per_image']  = 2
        images = h.create_dataset('images',(len(imgs),3,256,256), dtype='uint8')
        print('\nReading images and captions, storing to file...\n"')
        enc_captions = []
        caplens = []

        for i, img in enumerate(tqdm(imgs)):
            if len(img['final_captions']) < 2:
                print(img['image_path'])
                captions = img['final_captions'] + [choice(img['final_captions']) for _ in range(2 - len(img['final_captions']))]
            else:
                captions = sample(img['final_captions'], k=2)

            # Sanity check
            assert len(captions) == 2
            #read image
            for image_name in image_name_data:
                #print(img['image_path'].split('/')[-1], image_name.split('.')[0])
                if img['image_path'].split('/')[-1] == image_name.split('.')[0]:
                    img_path = img['image_path'] +'.'+ image_name.split('.')[-1]
            #print(img_path)
            IMG = np.asarray(Image.open(img_path))
            
            if len(IMG.shape) == 2:
                IMG = IMG[:, :, np.newaxis]
                IMG = np.concatenate([IMG, IMG, IMG], axis=2)
            
            if IMG.shape[-1] == 4:
                #print(IMG.shape)
                IMG = Image.open(img_path).convert("RGB") 
           
            IMG = imresize(IMG, (256, 256))
            IMG = IMG.transpose(2, 0, 1)
            assert IMG.shape == (3, 256, 256)
            assert np.max(IMG) <= 255
            
            images[i] = IMG
            
            #encode caption
            for sent in captions:
                # word --> word_map['word'] = 1, [1,2,3,4,5] + [0,0,0] 
                enc_c = [word_map.get(word, word_map['<unk>']) for word in sent] + [word_map['<pad>']] * (max_len - len(sent))
                # Find caption lengths
                c_len = len(sent)
                
                enc_captions.append(enc_c)
                caplens.append(c_len)
        # Sanity check
        print(images.shape[0])
        print(len(enc_captions), len(caplens))
        assert images.shape[0] * 2 == len(enc_captions) == len(caplens)

        # Save encoded captions and their lengths to JSON files
        with open(os.path.join(output_folder,  '{}_CAPTIONS'.format(split) + '.json'), 'w') as j:
            json.dump(enc_captions, j)

        with open(os.path.join(output_folder,  '{}_CAPLENS'.format(split) + '.json'), 'w') as j:
            json.dump(caplens, j)
            
#########################################
            
def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.
    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.
    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best):
    """
    Saves model checkpoint.
    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + 'news' + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


if __name__ == '__main__':
    annotation_root = './raw_data/新闻描述二期第二批数据/'
    image_root = './raw_data/images_2'
    
    #构造数据集
    annotation_file = [file for file in os.listdir(annotation_root) if file[-4:] == '.txt']
    data_annotations = []
    for files in annotation_file:
        for annotation in os.listdir(os.path.join(annotation_root,files)):
            entity = bulid_data(os.path.join(annotation_root,files),annotation)
            #print(image_name,label)
            if entity:
                data_annotations.append(entity)
   
    # 构造单词表        
    vocab,param = bulid_vocab(data_annotations)
    # 单词索引表
    itow = {i : w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
    wtoi = {w:i  for i,w in enumerate(vocab)} # inverse table
    # 保存单词表
    with open('./process_data_2/word_map.json','w') as f:
        json.dump(wtoi,f)
      
    for img in data_annotations:
        img['image_path'] = image_root +'/'+ img['image_name']
    
    train_data = data_annotations[:int(0.8*len(data_annotations))]
    val_data = data_annotations[int(0.8*len(data_annotations)):]
    #输出网络输入文件：编码caption
    create_input_files(train_data,'train',param, wtoi,image_root)
    create_input_files(val_data, 'val', param, wtoi,image_root) 
    #保存data_annotations
    with open('./process_data_2/data_annotations.json','w') as f:
        json.dump(data_annotations,f)
    
