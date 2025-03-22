如何修改Isaac Gym中的地形——基于legged_gym框架
**本工程相对于legged_gym原版改动过的.py文件如下**：
```python
legged_gym/envs/base/base_task.py

legged_gym/envs/base/legged_robot_config.py

legged_gym/envs/base/legged_robot.py

legged_gym/envs/go2/go2_config.py

legged_gym/envs/go2/go2_robot.py

legged_gym/envs/__init__.py

legged_gym/utils/terrain.py

legged_gym/utils/task_registry.py
```

**首先声明一下我修改该工程文件比较大的地方：**  
1.leggedgym的bug 在utils/terrain.py里面 
```python
def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
```
应该改成：
```python
def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.length_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
```

第二处在legged_robot_config和go2_config里面，有几个配置用的布尔量：
应该改成：
```python
  curriculum = True
  selected = False # select a unique terrain type and pass all arguments
```
被我全部删了，一点也不好用，我新加了一个字符变量terrain_generation_rule用来制定地形生成规则。


**下面正式开始：**
主要分为两个文件，`legged_robot.py` 和 `legged_robot_config.py`，前者记录的都是方法，后者则全部是默认的配置文件。

运行 `train.py`:

```python
from legged_gym.envs import 这一步会执行 envs 文件夹内的__init__.py
from legged_gym.utils import get_args, task_registry
import torch
```

envs 文件夹内的`__init__.py`:
```python
from legged_gym.utils.task_registry import task_registry

task_registry.register( "anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO() )
task_registry.register( "anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO() )
task_registry.register( "anymal_b", Anymal, AnymalBRoughCfg(), AnymalBRoughCfgPPO() )
task_registry.register( "a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO() )
task_registry.register( "cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO() )
task_registry.register( "go2", Go2Robot, Go2RoughCfg(), Go2RoughCfgPPO())
```

注册，task_classes 就是一个字典,存储了任务类

然后运行 train.py 里面的内容：

```python
def train(args):
env, env_cfg = task_registry.make_env(name=args.task, args=args)
ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
```
```python
env, env_cfg = task_registry.make_env(name=args.task, args=args)
```
首先,__init__.py 用 task_registry.register()函数注册了针对某个机器人的任务类,我们以 go2 为例:

```python
task_registry.register( "go2", Go2Robot, Go2RoughCfg(),Go2RoughCfgPPO())
'''
名字:"go2"(str) task_class: Go2Robot(VecEnv),
环境配置:Go2RoughCfg(LeggedRobotCfg) 
训练配置: Go2RoughCfgPPO(LeggedRobotCfgPPO)
'''
```

然后在 `make_env()`函数中,根据传入的任务名字,获取对应的任务类,环境配置和训练配置,并创建一个任务类的实例 `env`
具体来讲,先 `task_class = self.get_task_class(name)`,后面这个函数会返回前面用 `register`注册的类,也就是说 `task_class` 现在是一个类,
以 go2 为例,`task_class=Go2Robot`,也就是说这个时候 `task_class` 就是 `Go2Robot` 这个类了
后面的代码就是创建一个 `Go2Robot` 这个类的实例,并把它赋值给 `env`:

```python
env = task_class(cfg = env_cfg,
                       sim_params = sim_params,
                       physics_engine = args.physics_engine,
                       sim_device = args.sim_device,
                       headless = args.headless)
```

env 存储着关于该机器人的所有配置内容，具体来讲，env_cfg 就是对应的 LeggedRobotCfg 基类的子类，我们以 go2 为例，就是 Go2RoughCfg 类
task_class 会创建 Go2Robot 类，然后会执行 LeggedRobot 父类的构造函数，然后  `super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)`又会执行 BaseTask 的内容。
BaseTask 的构造函数里面唯一有用的是 `self.create_sim()`，在这里又会跳回到对应子类的 `create_sim()`函数,创建地形环境。

```python
  def create_sim(self):
  """
  train.py里面make_env函数最后a会创建一个Go2Robot类的实例,然后会在构造函数里面调用这个函数初始化环境
  """
  self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly

  self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
  mesh_type = self.cfg.terrain.mesh_type

  if mesh_type in ['heightfield', 'trimesh']:
  self.terrain = Terrain(self.cfg.terrain, self.num_envs)
  if mesh_type=='plane':
    self._create_ground_plane()
  elif mesh_type=='heightfield':
    self._create_heightfield()
  elif mesh_type=='trimesh':
    self._create_trimesh()
  elif mesh_type is not None:
  raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
  self._create_envs()
```
重头戏是第一步执行的这个函数：
```python
  self.terrain = Terrain(self.cfg.terrain, self.num_envs)
```
会创建一个Terrain类的实例terrain，这个Terrain类可不一般，这个是isaacgym官方专门生成地形用的接口。经过我的研究，类里面的对象和所有方法都是用在构造函数里面的，所以我们去看构造函数就OK了。  

**另外必须要注意的是，`self.cfg = cfg`相当于是加载了LeggedRobotCfg.terrain，也就是Go2RoughCfg.terrain里面我们配置的地形参数！！！！**
```python
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        '''
        self.cfg 是 Terrain.cfg 类的成员
        cfg 是 LeggedRobotCfg.terrain 类的基础配置，这个函数相当于解算了地形的配置参数
        '''
        self.num_robots = num_robots

        # 传入障碍配置参数
        self.cfg = cfg
        self.type = cfg.mesh_type

        if self.type in ["none", 'plane']: # 一共有四种地形类型，none, plane, trimesh, heightfield
            return
        
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width

        # 假设 cfg.terrain_proportions = [0.2, 0.3, 0.5]
        # 那么 self.proportions = [0.2, 0.5, 1.0]
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        # num_sub_terrains:总网格数量:整个地形被分成 num_rows × num_cols 个小区域。
        # p.s.实例变量（self.xxx）可以在运行时动态添加，
        # 即使 cfg（也就是 LeggedRobotCfg.terrain）在定义时没有 num_sub_terrains，也可以在实例上动态创建这个属性。
        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols

        # env_origins:存储每个地形区域的起点坐标，初始化为全零。
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        # 每一个小网格长/宽方向像素数(每一个像素坐标对应某处地形某一个点)=总宽度/地形的水平分辨率(1 像素代表多少米)
        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)
        self.border = int(cfg.border_size/self.cfg.horizontal_scale)

        # 地形行方向总像素=地形行数*每行像素数+2*边界像素
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        # 我们给每一个像素一个高度数据，所以先建立一个tot_rows行，tot_cols列的矩阵
        # 初始值全零，这个就是原始高度场，很重要，不要忘记这个变量！！！
        # 另外，这是个整形，就是因为byd isaacgym官方文档输入地形参数的接口必须给整形！！！
        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)

        if cfg.terrain_generation_rule == 'curriculum':
            self.curiculum()
        elif cfg.terrain_generation_rule == 'fixed':
            self.selected_terrain()
        elif cfg.terrain_generation_rule == 'random':    
            self.randomized_terrain() 
        elif cfg.terrain_generation_rule == 'obstest':    
            self.obstest_terrain()       

        # 如果地形类型是 "trimesh"，则调用 convert_heightfield_to_trimesh 函数
        # 将高度场转换为三角网格（trimesh），并存储相应的顶点和三角形数据。
        if self.type in ['trimesh']:
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(
                self.height_field_raw,
                self.cfg.horizontal_scale
                self.cfg.vertical_scale,
                self.cfg.slope_treshold
            )

```
不难发现，似乎整段看下来也没有设定任何一个像素的高度，那么这个工作是在哪里做的呢?其实是在这几个函数里面（当然其实还套了一层，我只能说这项目代码真鸡儿难读）

```python
    if cfg.terrain_generation_rule == 'curriculum':
        self.curiculum()
    elif cfg.terrain_generation_rule == 'fixed':
        self.selected_terrain()
    elif cfg.terrain_generation_rule == 'random':    
        self.randomized_terrain() 
    elif cfg.terrain_generation_rule == 'obstest':    
        self.obstest_terrain()      
```
那么我们来看看这三个函数里面都是什么东西，本质上来说，都是通过一些方法获取了difficulty和choice的值，然后用循环遍历每一个地形格子，循环里面使用make_terrain函数生成地形，所以我们必须再去看看make_terrain函数。

```python
    def randomized_terrain(self):
        '''
            坡度 (slope):difficulty 值越高，坡度越陡，地形就会更具挑战性。
            阶梯高度 (step_height):difficulty 值越高，阶梯的高度也会增大，增加训练的难度。
            障碍物高度 (discrete_obstacles_height):difficulty 值越高，障碍物的高度也会增加。
            间隔 (gap_size):difficulty 还会影响生成的“空隙”或“坑”的大小和深度，增加训练的难度。
        '''   
        for k in range(self.cfg.num_sub_terrains): # for(int k = 0; k < self.cfg.num_sub_terrains; k++)
            # 遍历每一个格子
            # 给每一个格子编号，编号范围是 0 ~ num_sub_terrains
            '''
            np.unravel_index(k, (num_rows, num_cols)) 将 1D 索引 k 转换为 2D 索引 (i, j)，表示该地形在整张地图中的行列编号。
            示例: 假设 num_rows=3, num_cols=4，那么 num_sub_terrains=3×4=12，编号 k 依次是 0~11。
            np.unravel_index 的转换如下:
            k=0  → (0,0)   k=1  → (0,1)   k=2  → (0,2)   k=3  → (0,3)
            k=4  → (1,0)   k=5  → (1,1)   k=6  → (1,2)   k=7  → (1,3)
            k=8  → (2,0)   k=9  → (2,1)   k=10 → (2,2)   k=11 → (2,3)
            '''
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            # np.random.uniform(0, 1) 生成一个 0~1 之间的浮点数，用于随机选择地形类型。
            choice = np.random.uniform(0, 1) 

            # 从 [0.5, 0.75, 0.9] 这三个值中随机选择一个,随机选择地形难度
            difficulty = np.random.choice([0.5, 0.75, 0.9])

            # 给每一个具体的网格生成地形。
            terrain = self.make_terrain(choice, difficulty)
            
            # 把生成的地形添加到地图中。
            self.add_terrain_to_map(terrain, i, j) 
        
    def curiculum(self):
        '''
            curiculum 方法生成基于课程的地形。难度随着行数增加而逐渐增大，
            choice 根据列数进行选择。同样，生成对应的地形并将其添加到地图中。
        '''
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                
                # 根据行列给出不同的难度和选择值，行列越大，难度越大，选择值越大
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                # 有难度和选择值给每一个具体的网格生成地形。
                terrain = self.make_terrain(choice, difficulty)

                # 把生成的地形添加到地图中。
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
```
这里是真正的重头戏，别眨眼！
```python
    def make_terrain(self, choice, difficulty):
        '''
          给每一个具体的网格生成地形。
          make_terrain 方法根据传入的 choice 和 difficulty 生成地形。
          该方法首先创建一个 SubTerrain 对象，然后根据 choice 和 difficulty 生成不同类型的地形。
        '''

        # p.s.这里为什么要创建一个SubTerrain对象呢？因为要用isaacgym官方封装的库函数(terrain_utils.py)
        # 来生成地形。所以想调用这些函数就必须先创建一个 SubTerrain 对象。
        terrain = terrain_utils.SubTerrain(
                                terrain_name     = "terrain",
                                width            = self.width_per_env_pixels,
                                length           = self.width_per_env_pixels,
                                vertical_scale   = self.cfg.vertical_scale,
                                horizontal_scale = self.cfg.horizontal_scale)
        ########################################################################################
        ########################################################################################
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        
        '''
          slope:倾斜度，随着难度增加，倾斜度逐渐加大。
          step_height:步高，随着难度增加，步高增加。
          discrete_obstacles_height:障碍物的高度。
          stepping_stones_size:跳石的大小，随着难度增加，跳石的大小减小。
          stone_distance:跳石间距，难度为 0 时较小，其他难度较大。
          gap_size:间隙大小，随着难度增加而增大。
          pit_depth:坑深，随着难度增加，坑深增加。
        '''
        # choice 用来决定生成哪种类型的地形。它的值与 self.proportions 的数组进行比较，
        # 从而选择不同的地形生成方法。每个 choice 范围对应一种特定的地形类型。
        # 前置：假设LeggedRobotCfg.terrain.terrain_proportions = [0.1,   0.1,    0.35, 0.25, 0.2]
        # 各种地形类别占比：                                      平滑斜坡 崎岖斜坡 上楼梯 下楼梯 离散地形
        # 那么 执行该函数之前这个值会被传递并且改造成self.proportions = [0.1, 0.2, 0.55, 0.8, 1.0]（改成累加式）

        # 斜坡
        if choice < self.proportions[0]:
            # choice < self.proportions[0] / 2 则反转斜坡的方向（通过 slope *= -1）下坡
            # self.proportions[0]/2 < choice < self.proportions[0] 上坡
            if choice < self.proportions[0] / 2.0:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        
        # 崎岖斜坡
        elif choice < self.proportions[1]:
            # self.proportions[0] < choice < self.proportions[1]
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.) # 生成斜坡地形
            # 增加随机噪声
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        
        # 上下樓梯
        elif choice < self.proportions[3]:
            if choice < self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        
        # 生成离散地形（石板地形）
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        '''        
        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        '''       
        return terrain
```
make_terrain函数接受difficulty和choice两个参数，根据不同的choice生成不同类型的地形，而difficulty则决定地形的难度（such as 更高的台阶 更陡的坡）
具体地形的比例和类型，其实都是可以自己改的。比例在cfg类里面改，地形的话，isaacgym安装文件夹里面的terrain.py还有很多其他的地形可以直接用，当然你也可以自己写。

**我们这里还需要明确一个问题，最后这些函数修改的都是`height_field_raw`这个变量。**

这里再提一下`self.add_terrain_to_map(terrain, i, j)`是干嘛的：
```python
    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        # start_x, end_x：通过 i（行号）计算地形在 x 方向的起始和结束位置，单位是像素。
        # self.length_per_env_pixels 代表每个網格的宽度，self.border 是地图的边界偏移量
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels

        # 将生成的地形高度数据添加到地图
        self.height_field_raw[start_x: end_x, start_y : end_y] = terrain.height_field_raw

        # env_origin_x 和 env_origin_y：计算环境的原点位置（中心点），
        # 分别基于行 i 和列 j 的位置。self.env_length 和 self.env_width 分别表示每个环境的长度和宽度。
        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width

        # x1, x2, y1, y2：这些是通过环境的大小、水平缩放比例来确定的索引，
        # 表示环境中心区域的范围。这些索引用于在 terrain.height_field_raw 中查找环境的最大高度。
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width /2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width /2. + 1) / terrain.horizontal_scale)

        # 通过在 terrain.height_field_raw 中查找环境中心区域的最大高度来计算该环境的 z 坐标（高度），
        # 然后将其乘以 vertical_scale 来进行缩放。
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale

        # 最后，将计算出的环境原点的坐标 [env_origin_x, env_origin_y, env_origin_z] 
        # 存储到 self.env_origins 数组中，表示该地形在整个地图上的位置。
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
```
`self.height_field_raw[start_x: end_x, start_y : end_y] = terrain.height_field_raw`: 第i行j列的格子对应的像素是`[start_x: end_x], [start_y : end_y]`，把这里的地形数据写到记录所有地形高度的张量`self.height_field_raw`里面
同时记录一下每个地形的中点坐标和最高点坐标，当然其实不用也可以。**请记住env_origins这个变量，他非常重要，记录的是机器人初始刷新的位置坐标！！！**


最后有了地形高度数组，如果是highfield类型就到此位置，但是我们一定要用trimesh类的地形，所以要做额外的处理。
```python
        # 如果地形类型是 "trimesh"，则调用 convert_heightfield_to_trimesh 函数
        # 将高度场转换为三角网格（trimesh），并存储相应的顶点和三角形数据。
        if self.type in ['trimesh']:
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(
                self.height_field_raw,
                self.cfg.horizontal_scale,
                self.cfg.vertical_scale,
                self.cfg.slope_treshold
            )
```
这个时候地形数据就在`self.vertices, self.triangles`里面了。这样我们的Terrain类就创建完毕了，剩下需要做的就是把地形数据写进isaacgym里面了！
我想着你也忘了最前面创建完Terrain类以后干嘛了，所以我们再粘贴一下：
```python
    def create_sim(self): 
        """ 
          train.py里面make_env函数最后a会创建一个Go2Robot类的实例,然后会在构造函数里面调用这个函数初始化环境
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type

        if mesh_type in ['heightfield', 'trimesh']:
            # Terrain类在这里
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()  
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()
```
让我们看看这几个函数都在整什么幺蛾子，平地的我们就不看了，从高度场开始（虽然我们一般用trimesh）：

```python
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        # 创建gymapi的类型
        hf_params = gymapi.HeightFieldParams()

        # 这下面都是载入参数
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        # 调用isaacgym的api接口把地形写进去，注意self.terrain.height_field_raw这个输入参数，原文写的是highsamples，我认为是严重错误的
        self.gym.add_heightfield(self.sim, self.terrain.height_field_raw, hf_params)
        self.height_samples = torch.tensor(self.terrain.height_field_raw).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
```

```python
    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        # 创建gymapi的类型
        tm_params = gymapi.TriangleMeshParams()

        # 这下面都是载入参数
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution

        # 调用isaacgym的api接口把地形写进去，注意self.terrain.height_field_raw这个输入参数，原文写的是highsamples，我认为是严重错误的
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.height_field_raw).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
```

发现了吧，其实都是载入参数的活，不需要再多改什么了。
那么恭喜你，所有和地形有关系的部分**全部结束辣！！！**，那么问题来了，我们应该怎么改？

梳理一下地形生成逻辑：创建机器人类和cfg类->构造函数里面执行`create_sim()`->读取cfg类的配置信息，创建一个Terrain类的实例->Terrain类的构造函数计算基础配置信息

->根据terrain_generation_rule选择不同的地形生成规则->最后地形存储在height_field_raw里面->用_create_trimesh（heightfield）写入gym环境中。

所以我们需要的是修改这一步：**根据terrain_generation_rule选择不同的地形生成规则**:utils/terrain.py里面
```python
      if cfg.terrain_generation_rule == 'curriculum':
          self.curiculum()
      elif cfg.terrain_generation_rule == 'fixed':
          self.selected_terrain()
      elif cfg.terrain_generation_rule == 'random':    
          self.randomized_terrain() 
      elif cfg.terrain_generation_rule == 'obstest':    
          self.obstest_terrain()      
```
在这里加上你想要的地形生成规则，内部怎么执行自己写就好，总之就是给每一个格子赋不同的高度场值，具体怎么写可以看我写的`obstest_terrain()`，很简单的！

```python
    def obstest_terrain(self):
        # 这个方法的输入参数很重要，我们需要生成地形的长度和宽度，这决定了返回的height_field_raw数组大小。
        def new_sub_terrain(length = self.terrain_length, width = self.terrain_width): 
            '''
              定义 new_sub_terrain 方法
              new_sub_terrain 是一个辅助函数，创建一个新的 SubTerrain 对象
            '''
           
            return SubTerrain(terrain_name     = "terrain",
                              width            = width,
                              length           = length,
                              vertical_scale   = self.cfg.vertical_scale,
                              horizontal_scale = self.cfg.horizontal_scale)
        '''
        列col/length
        ^
        |
        |
        |
        |
        |
        |
        |
        |
        |
        |
        |
        |
        |
        |---------------------------------------------------->行row/width
        
        按照不同列生成不同的地形，那就需要在每一组地形块的所有行生成不同的地形
        length_per_env_pixels:每一个地形块的行像素数
        width_per_env_pixels :每一个地形块的列像素数
        tot_rows:总行像素数
        tot_cols:总列像素数
        '''
        # 平坦地形
        self.height_field_raw[0*self.length_per_env_pixels:1*self.length_per_env_pixels,:] = (
            sloped_terrain(new_sub_terrain(width=self.length_per_env_pixels,length = self.tot_rows), slope = 0.0).height_field_raw
        )
        
        # 上坡地形
        self.height_field_raw[1*self.length_per_env_pixels:2*self.length_per_env_pixels,:] = (
            pyramid_sloped_terrain(new_sub_terrain(width=self.length_per_env_pixels,length = self.tot_rows), slope = -0.3).height_field_raw
        )
        
        # 随机地形
        self.height_field_raw[2*self.length_per_env_pixels:3*self.length_per_env_pixels,:] = (
             random_uniform_terrain(new_sub_terrain(width=self.length_per_env_pixels,length = self.tot_rows), 
                                    min_height=-0.15, max_height=0.15, 
                                    step=0.2, downsampled_scale=0.5).height_field_raw
        )
        
        # 带障碍物的地形
        self.height_field_raw[3*self.length_per_env_pixels:4*self.length_per_env_pixels,:] = (
             discrete_obstacles_terrain(new_sub_terrain(width=self.length_per_env_pixels,length = self.tot_rows), 
                                        max_height=0.15, min_size=1., max_size=5., num_rects=20).height_field_raw
        )
        
        # 波浪地形
        self.height_field_raw[4*self.length_per_env_pixels:5*self.length_per_env_pixels,:] = (
             wave_terrain(new_sub_terrain(width=self.length_per_env_pixels,length = self.tot_rows), 
                          num_waves=2., amplitude=1.).height_field_raw
        )
        
        # 楼梯地形
        self.height_field_raw[5*self.length_per_env_pixels:6*self.length_per_env_pixels,:] = (
             stairs_terrain(new_sub_terrain(width=self.length_per_env_pixels,length = self.tot_rows), 
                            step_width=0.75, step_height=0.25).height_field_raw
        )
        
        # 楼梯地形
        self.height_field_raw[6*self.length_per_env_pixels:7*self.length_per_env_pixels,:] = (
             stairs_terrain(new_sub_terrain(width=self.length_per_env_pixels,length = self.tot_rows), 
                            step_width=0.75, step_height=0.25).height_field_raw
        )
                
        # 跳石地形
        self.height_field_raw[7*self.length_per_env_pixels:8*self.length_per_env_pixels,:] = (
             stepping_stones_terrain(new_sub_terrain(width=self.length_per_env_pixels,length = self.tot_rows), 
                                     stone_size=1.,stone_distance=0.25, 
                                     max_height=0.2, platform_size=0.).height_field_raw
        )
```

到此为止其实还没有完全解决问题，我们还需要设置机器人reset时候重新刷新的位置。

```python
    def create_sim(self): 
        """ 
          train.py里面make_env函数最后a会创建一个Go2Robot类的实例,然后会在构造函数里面调用这个函数初始化环境
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type

        if mesh_type in ['heightfield', 'trimesh']:
            # Terrain类在这里
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()  
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()
```
看一下这个_create_envs()：

```python
def _create_envs(self):
        '''省略了一部分'''
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)

        # 创建一个 Transform 对象，表示机器人的初始位姿
        start_pose = gymapi.Transform()

        # 设置初始位置，将 self.base_init_state 的前三个值 (x, y, z) 赋给 start_pose.p
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        # 调用 `_get_env_origins()` 方法，获取环境原点信息
        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # 创建环境实例
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))

            # 获取该环境的原点位置，并克隆一个副本
            pos = self.env_origins[i].clone()

            # 在 x 和 y 方向上随机偏移 [-1, 1] 范围内的值
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
```

设置起始位置所通过读取_get_env_origins方法,其实坐标记录在self.env_origins变量里面，所以我们只需要对_get_env_origins稍作修改就好。
注意注意，self.env_origins变量单位是米，不是像素！！！
