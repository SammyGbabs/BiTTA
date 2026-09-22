
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import conf

opt = conf.CassavaOpt


class CassavaDataset(torch.utils.data.Dataset):
    def __init__(self, file='', domains=None, activities=None, max_source=100, transform='none'):
        self.domains = domains
        self.max_source = max_source
        self.img_shape = opt['img_size']
        self.file_path = opt['file_path']
        assert set(domains).issubset(set(opt['domains'] + ['test']))
        if domains[0] == "test":
            self.domains[0] = opt['src_domains'][0]
        self.sub_paths = self.domains

        if transform == 'src':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif transform == 'val':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            raise NotImplementedError
        self.preprocessing()

    def preprocessing(self):
        self.dataset = []
        self.domain_list = []
        for i, sub_path in enumerate(self.sub_paths):
            path = f'{self.file_path}/{sub_path}/'
            dataset = datasets.ImageFolder(path, transform=self.transform)
            self.dataset += [dataset]
            self.domain_list += ([i] * len(dataset))
        self.dataset = torch.utils.data.ConcatDataset(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def get_num_domains(self):
        return len(self.domains)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        img, cl = self.dataset[idx]
        dl = self.domain_list[idx]
        return img, torch.tensor([cl]), torch.tensor([dl])
