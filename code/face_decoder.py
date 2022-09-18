class ArcFaceRescale(nn.Module):
    def __init__(self, in_features, out_features, scale, false_scale, margin, m1,m3, with_theta, clip_thresh,
                 clip_value):
        super(ArcFaceRescale, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.false_scale = false_scale
        self.margin = margin
        self.with_theta = with_theta
        self.clip_thresh = clip_thresh
        self.clip_value = clip_value
        self.m1 = m1
        self.m3 = m3

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

        self.mm = math.sin(math.pi - self.margin) * self.margin
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, label, curr_step=-1):
        ex = input / torch.norm(input, 2, 1, keepdim=True)
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        cos = torch.mm(ex, ew.t())
        src_logits = cos.detach().clone()
        a = torch.zeros_like(cos)
        b = torch.ones_like(cos)
        c = torch.zeros_like(cos)
        thetas = []

        scale_matrix = torch.ones_like(cos) * self.false_scale
        for i in range(a.size(0)):
            lb = int(label[i])
            scale_matrix[i, lb] = self.scale

        for i in range(a.size(0)):
            lb = int(label[i])
            if cos[i, lb].item() > self.clip_thresh:
                cos[i, lb] = self.clip_value

        for i in range(a.size(0)):
            lb = int(label[i])
            a[i, lb] = a[i, lb] + self.margin
            b[i, lb] = b[i, lb] * self.m1
            c[i, lb] = c[i, lb] - self.m3

        return scale_matrix *  (torch.cos(b*torch.acos(cos) + a)+c)

        
        




