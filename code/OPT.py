import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau
from typing import Dict, Any


class OptimizerFactory:
    """优化器工厂类"""
    
    @staticmethod
    def create_optimizer(model_parameters, optimizer_config: Dict[str, Any]):
        """
        创建优化器
        Args:
            model_parameters: 模型参数
            optimizer_config: 优化器配置
        Returns:
            optimizer: 优化器实例
        """
        optimizer_type = optimizer_config.get('type', 'sgd').lower()
        learning_rate = optimizer_config.get('learning_rate', 0.001)
        weight_decay = optimizer_config.get('weight_decay', 0.0005)
        
        if optimizer_type == 'sgd':
            momentum = optimizer_config.get('momentum', 0.9)
            optimizer = optim.SGD(
                model_parameters,
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adam':
            betas = optimizer_config.get('betas', (0.9, 0.999))
            eps = optimizer_config.get('eps', 1e-8)
            optimizer = optim.Adam(
                model_parameters,
                lr=learning_rate,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adamw':
            betas = optimizer_config.get('betas', (0.9, 0.999))
            eps = optimizer_config.get('eps', 1e-8)
            optimizer = optim.AdamW(
                model_parameters,
                lr=learning_rate,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'rmsprop':
            alpha = optimizer_config.get('alpha', 0.99)
            eps = optimizer_config.get('eps', 1e-8)
            momentum = optimizer_config.get('momentum', 0)
            optimizer = optim.RMSprop(
                model_parameters,
                lr=learning_rate,
                alpha=alpha,
                eps=eps,
                weight_decay=weight_decay,
                momentum=momentum
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        return optimizer
    
    @staticmethod
    def create_scheduler(optimizer, scheduler_config: Dict[str, Any]):
        """
        创建学习率调度器
        Args:
            optimizer: 优化器
            scheduler_config: 调度器配置
        Returns:
            scheduler: 学习率调度器
        """
        scheduler_type = scheduler_config.get('type', 'step').lower()
        
        if scheduler_type == 'step':
            step_size = scheduler_config.get('step_size', 30)
            gamma = scheduler_config.get('gamma', 0.1)
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        
        elif scheduler_type == 'multistep':
            milestones = scheduler_config.get('milestones', [60, 90])
            gamma = scheduler_config.get('gamma', 0.1)
            scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        
        elif scheduler_type == 'cosine':
            T_max = scheduler_config.get('T_max', 100)
            eta_min = scheduler_config.get('eta_min', 0)
            scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        
        elif scheduler_type == 'plateau':
            mode = scheduler_config.get('mode', 'min')
            factor = scheduler_config.get('factor', 0.1)
            patience = scheduler_config.get('patience', 10)
            threshold = scheduler_config.get('threshold', 1e-4)
            scheduler = ReduceLROnPlateau(
                optimizer, mode=mode, factor=factor, 
                patience=patience, threshold=threshold
            )
        
        elif scheduler_type == 'none':
            scheduler = None
        
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
        return scheduler


class TrainingOptimizer:
    """训练优化器管理类"""
    
    def __init__(self, model, optimizer_config: Dict[str, Any], scheduler_config: Dict[str, Any] = None):
        """
        初始化训练优化器
        Args:
            model: 模型
            optimizer_config: 优化器配置
            scheduler_config: 调度器配置
        """
        self.model = model
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config or {'type': 'none'}
        
        # 创建优化器
        self.optimizer = OptimizerFactory.create_optimizer(
            model.parameters(), optimizer_config
        )
        
        # 创建学习率调度器
        self.scheduler = OptimizerFactory.create_scheduler(
            self.optimizer, self.scheduler_config
        )
        
        self.current_lr = self.get_current_lr()
    
    def step(self, loss=None):
        """执行一步优化"""
        self.optimizer.step()
        
        # 更新学习率
        if self.scheduler is not None:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                if loss is not None:
                    self.scheduler.step(loss)
            else:
                self.scheduler.step()
        
        self.current_lr = self.get_current_lr()
    
    def zero_grad(self):
        """清零梯度"""
        self.optimizer.zero_grad()
    
    def get_current_lr(self):
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']
    
    def state_dict(self):
        """获取状态字典"""
        state = {
            'optimizer': self.optimizer.state_dict(),
            'optimizer_config': self.optimizer_config,
            'scheduler_config': self.scheduler_config
        }
        if self.scheduler is not None:
            state['scheduler'] = self.scheduler.state_dict()
        return state
    
    def load_state_dict(self, state_dict):
        """加载状态字典"""
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if self.scheduler is not None and 'scheduler' in state_dict:
            self.scheduler.load_state_dict(state_dict['scheduler'])


def get_default_optimizer_config():
    """获取默认优化器配置"""
    return {
        'type': 'sgd',
        'learning_rate': 0.001,
        'momentum': 0.9,
        'weight_decay': 0.0005
    }


def get_default_scheduler_config():
    """获取默认调度器配置"""
    return {
        'type': 'multistep',
        'milestones': [60, 90],
        'gamma': 0.1
    }


def create_yolo_optimizer(model, learning_rate=0.001, weight_decay=0.0005):
    """
    创建适用于 YOLO 的优化器
    Args:
        model: YOLO 模型
        learning_rate: 学习率
        weight_decay: 权重衰减
    Returns:
        training_optimizer: 训练优化器
    """
    optimizer_config = {
        'type': 'sgd',
        'learning_rate': learning_rate,
        'momentum': 0.9,
        'weight_decay': weight_decay
    }
    
    scheduler_config = {
        'type': 'multistep',
        'milestones': [60, 90],
        'gamma': 0.1
    }
    
    return TrainingOptimizer(model, optimizer_config, scheduler_config)


def create_adam_optimizer(model, learning_rate=0.001, weight_decay=0.0005):
    """
    创建 Adam 优化器
    Args:
        model: 模型
        learning_rate: 学习率
        weight_decay: 权重衰减
    Returns:
        training_optimizer: 训练优化器
    """
    optimizer_config = {
        'type': 'adam',
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'betas': (0.9, 0.999),
        'eps': 1e-8
    }
    
    scheduler_config = {
        'type': 'cosine',
        'T_max': 100,
        'eta_min': 1e-6
    }
    
    return TrainingOptimizer(model, optimizer_config, scheduler_config)


def test_optimizer():
    """测试优化器"""
    import torch.nn as nn
    
    # 创建简单模型
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )
    
    print("Testing YOLO optimizer...")
    yolo_opt = create_yolo_optimizer(model)
    print(f"Initial LR: {yolo_opt.get_current_lr():.6f}")
    
    # 模拟训练步骤
    for epoch in range(5):
        yolo_opt.zero_grad()
        # 这里会有实际的前向传播和反向传播
        yolo_opt.step()
        print(f"Epoch {epoch+1}, LR: {yolo_opt.get_current_lr():.6f}")
    
    print("\nTesting Adam optimizer...")
    adam_opt = create_adam_optimizer(model)
    print(f"Initial LR: {adam_opt.get_current_lr():.6f}")
    
    # 模拟训练步骤
    for epoch in range(5):
        adam_opt.zero_grad()
        adam_opt.step()
        print(f"Epoch {epoch+1}, LR: {adam_opt.get_current_lr():.6f}")
    
    print("Optimizer test completed!")


if __name__ == "__main__":
    test_optimizer()
