# robosuite exec process

获取到delta action后，先进入set_goal根据delta_action设定目标，在set_goal一开始会用self.update()更新状态信息，ee_pos发生变化，之后进入controller.run_controller()，这里也会update()，然后根据前面设定的goal去计算应该施加的力矩，最后在control的self.sim.data.ctrl[self._ref_joint_actuator_indexes] = self.torques这一步实现控制