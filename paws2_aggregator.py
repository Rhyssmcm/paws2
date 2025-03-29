import numpy as np
import csv
import rclpy
import pybullet as p
import pybullet_data

from pathlib import Path
from typing import Iterator, Sequence, Self
from dataclasses import dataclass, asdict

from sensor_msgs.msg import Imu as ImuMsg, JointState as JointStateMsg
from ros2_custom_msg.msg import FootPositions as FootPositionsMsg, JacobianMatrix as JacobianMatrixMsg
from rclpy.node import Node


HEADER = ['qx', 'qy', 'qz', 'qw',
         'ax', 'ay', 'az',
         'wx', 'wy', 'wz',
         # Joint angles
         'fr_hip', 'fr_knee', 'fr_foot',
         'fl_hip', 'fl_knee', 'fl_foot',
         'rr_hip', 'rr_knee', 'rr_foot',
         'rl_hip', 'rl_knee', 'rl_foot',
         # Foot positions
         'fr_x', 'fr_y', 'fr_z',
         'fl_x', 'fl_y', 'fl_z',
         'rr_x', 'rr_y', 'rr_z',
         'rl_x', 'rl_y', 'rl_z']

IMU_TOPIC = '/imu/data'
JOINT_STATE_TOPIC = '/joint_states'
FEET_POSITIONS_TOPIC = '/foot_positions'
JACOBIAN_MATRIX_TOPIC = '/j_matrix'
OUT_FILE = 'data.csv'


#region DATACLASSES
@dataclass
class ImuQuaternion:
    """Quaternion data."""
    x: float
    y: float
    z: float
    w: float

    @property
    def np_array(self):
        return np.array([self.x, self.y, self.z, self.w])
    
    @classmethod
    def from_msg(cls, msg):
        """Create an instance from a message."""
        return cls(
            x=msg.orientation.x,
            y=msg.orientation.y,
            z=msg.orientation.z,
            w=msg.orientation.w,
        )
    
    def flatten(self) -> tuple:
        """Flatten the angles into a single array."""
        return self.x, self.y, self.z
    
    @classmethod
    def iterate(cls, seq: Sequence[Self]) -> Iterator[tuple]:
        """Flatten a sequence of foot positions into a single array."""
        for quaternion in seq:
            yield quaternion.flatten()
    

@dataclass
class ImuAngularVelocity:
    """Angular velocity data."""
    x: float
    y: float
    z: float

    @property
    def np_array(self):
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_msg(cls, msg: ImuMsg):
        """Create an instance from a message."""
        return cls(
            x=msg.angular_velocity.x,
            y=msg.angular_velocity.y,
            z=msg.angular_velocity.z,
        )
    
    def flatten(self) -> tuple:
        """Flatten the angles into a single array."""
        return self.x, self.y, self.z
    
    @classmethod
    def iterate(cls, seq: Sequence[Self]) -> Iterator[tuple]:
        """Flatten a sequence of foot positions into a single array."""
        for angular_velocity in seq:
            yield angular_velocity.flatten()
    

@dataclass
class ImuLinearAcceleration:
    """Linear acceleration data."""
    x: float
    y: float
    z: float

    @property
    def np_array(self):
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_msg(cls, msg: ImuMsg):
        """Create an instance from a message."""
        return cls(
            x=msg.linear_acceleration.x,
            y=msg.linear_acceleration.y,
            z=msg.linear_acceleration.z,
        )
    
    def flatten(self) -> tuple:
        """Flatten the angles into a single array."""
        return self.x, self.y, self.z
    
    @classmethod
    def iterate(cls, seq: Sequence[Self]) -> Iterator[tuple]:
        """Flatten a sequence of foot positions into a single array."""
        for linear_acceleration in seq:
            yield linear_acceleration.flatten()
    

@dataclass
class LegAngles:
    """Joint state data."""
    hip: float
    knee: float
    foot: float

    @classmethod
    def from_sequence(cls, seq: Sequence[float]):
        """Create an instance from a message."""
        if len(seq) != 3:
            raise ValueError("Sequence must have exactly 3 elements")
        
        return cls(
            hip=seq[0],
            knee=seq[1],
            foot=seq[2],
        )
    
    def flatten(self) -> tuple:
        """Flatten the angles into a single array."""
        return self.hip, self.knee, self.foot


@dataclass
class LegSetAngles:
    """Joint state data for a set of legs."""
    fr: LegAngles
    fl: LegAngles
    rr: LegAngles
    rl: LegAngles
    
    @classmethod
    def from_msg(cls, msg: JointStateMsg):
        """Create an instance from a message."""
        if len(msg.position) != 16:
            raise ValueError(f"Sequence must have exactly 16 elements."
                             " Got {len(msg.position)} instead.")
        
        return cls(
            fr=LegAngles.from_sequence(msg.position[0:3]),
            fl=LegAngles.from_sequence(msg.position[4:7]),
            rr=LegAngles.from_sequence(msg.position[8:11]),
            rl=LegAngles.from_sequence(msg.position[12:15]),
        )
    
    def flatten(self) -> tuple:
        """Flatten the foot positions into a single array."""
        return *self.fr, *self.fl, *self.rr, *self.rl
    
    @classmethod
    def iterate(cls, seq: Sequence[LegAngles]) -> Iterator[tuple]:
        """Flatten a sequence of foot positions into a single array."""
        for foot_position in seq:
            yield foot_position.flatten()
    

@dataclass
class FootPosition:
    """Foot position data."""
    x: float
    y: float
    z: float

    @classmethod
    def from_msg_foot(cls, msg_foot):
        """Create an instance from a message."""
        return cls(
            x=msg_foot.x,
            y=msg_foot.y,
            z=msg_foot.z,
        )
    

@dataclass
class FootSetPositions:
    """Foot position data for a set of legs."""
    fr: FootPosition
    fl: FootPosition
    rr: FootPosition
    rl: FootPosition
    
    @classmethod
    def from_msg(cls, msg: FootPositionsMsg):
        """Create an instance from a message."""
        return cls(
            fr=FootPosition.from_msg_foot(msg.fr_foot),
            fl=FootPosition.from_msg_foot(msg.fl_foot),
            rr=FootPosition.from_msg_foot(msg.rr_foot),
            rl=FootPosition.from_msg_foot(msg.rl_foot),
        )
    
    def flatten(self) -> tuple:
        """Flatten the foot positions into a single array."""
        return *self.fr, *self.fl, *self.rr, *self.rl
    
    @classmethod
    def iterate(cls, seq: Sequence[FootPosition]) -> Iterator[tuple]:
        """Flatten a sequence of foot positions into a single array."""
        for foot_position in seq:
            yield foot_position.flatten()
    

@dataclass
class JMatricesFeet:
    fr: np.ndarray
    fl: np.ndarray
    rr: np.ndarray
    rl: np.ndarray

    @classmethod
    def from_msg(cls, msg: JacobianMatrixMsg):
        """Create an instance from a message."""
        return cls(
            fr=np.array(msg.j_fr),
            fl=np.array(msg.j_fl),
            rr=np.array(msg.j_rr),
            rl=np.array(msg.j_rl),
        )
    
    
    def save_j_matrix(self, file_name: str, output_path: Path) :
        """Save the Jacobian matrices to a file."""
        if output_path.is_dir():
            _output_path = output_path / file_name
        else:
            _output_path = Path(output_path)
        
        with open(_output_path, 'w') as f:
            np.savetxt(f, self.fr, delimiter=',')
            np.savetxt(f, self.fl, delimiter=',')
            np.savetxt(f, self.rr, delimiter=',')
            np.savetxt(f, self.rl, delimiter=',')
#endregion DATACLASSES


class Paws2DataAggregator(Node):

    def __init__(self):
        # Stored data
        self.quaternion_data: list[ImuQuaternion] = []
        self.angular_velocity_data: list[ImuAngularVelocity] = []
        self.linear_acceleration_data: list[ImuLinearAcceleration] = []
        self.leg_set_angles_list: list[LegSetAngles] = []
        self.foot_positions: list[FootSetPositions] = []
        self.j_matrices_feet: list[JMatricesFeet] = []

    def subscribe(self, name="data_aggregator"):
        physicsClient = p.connect(p.SHARED_MEMORY)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Subscriptions
        self.imu_data_sub = self.node.create_subscription(
            ImuMsg, IMU_TOPIC, self.imu_callback, 10)
        self.joint_pos_sub = self.node.create_subscription(
            JointStateMsg, '/joint_states', self.joint_state_callback, 10)
        self.foot_pos_sub = self.node.create_subscription(
            FootPositionsMsg, '/foot_positions', self.foot_position_callback, 10)
        self.J_matrix_sub = self.node.create_subscription(
            JacobianMatrixMsg, '/j_matrix', self.j_matrix_callback, 10)
        

    def timer_callback(self):
        self.get_logger().info("Timer triggered, processing IMU data...")
        self.process_imu_data()

        
    def get_split_leg_angles(self) -> tuple[LegAngles, LegAngles, LegAngles, LegAngles]:
        """
        Usage:
            fr, fl, rr, rl = obj.get_leg_set_angles()
        """
        return zip(*self.leg_set_angles_list)
    
    def iterate(self) -> Iterator[tuple]:
        """Returns a snapshot of the deataset at any moment."""
        for entry in zip(
            ImuQuaternion.iterate(self.quaternion_data),
            ImuAngularVelocity.iterate(self.angular_velocity_data),
            ImuLinearAcceleration.iterate(self.linear_acceleration_data),
            LegSetAngles.iterate(self.leg_set_angles_list),
            FootSetPositions.iterate(self.foot_positions),
            JMatricesFeet.iterate(self.j_matrices_feet),
        ):
            yield tuple(entry)

#region   CALLBACKS
    def imu_callback(self, msg: ImuMsg):
        self.quaternion_data.append(ImuQuaternion.from_msg(msg))
        self.angular_velocity_data.append(ImuLinearAcceleration.from_msg(msg))
        self.linear_acceleration_data.append(ImuLinearAcceleration.from_msg(msg))

    def joint_state_callback(self, msg: JointStateMsg):
        leg_state = LegSetAngles.from_msg(msg)
        self.joint_states.append(leg_state)

    def foot_position_callback(self, msg: FootPositionsMsg):
        foot_positions = FootSetPositions.from_msg(msg)
        self.foot_positions.append(foot_positions)

    def j_matrix_callback(self, msg: JacobianMatrixMsg):
        j_matrices = JMatricesFeet.from_msg(msg)
        self.j_matrices_feet.append(j_matrices)
#endregion   CALLBACKS

if __name__ == '__main__':

    rclpy.init(args=None)
    
    node = Paws2DataAggregator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    rclpy.spin(node)
    node.subscribe()


    with open(OUT_FILE, 'a') as csvfile:
        writer = csv.writer(csvfile, header=None)
        writer.writerow(HEADER)
        for index, row in enumerate(node.iterate()):
            # Ignore jabocbian matrices for now
            writer.writerow(row[:33])
            j_matrix: JMatricesFeet = row[-1]
            j_matrix.save_j_matrix(f'{index}.csv', Path('j_matrices'))





# # # qx, qy, qz, qw, ax, ay, az, wx, wy, wz
# # collected_data = list(zip(*aggr.quaternion_data,
# #                           *aggr.angular_velocity_data,
# #                           *aggr.linear_acceleration_data))

# def myfunction(value1: str, options: list):
#     other(value1, *options)

# def other(value1, use_complete_mode, logging_level, url):
#     pass

# options = [False, 0, 'http://localhost:8080']
# myfunction('value1', options)
