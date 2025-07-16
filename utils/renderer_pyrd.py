# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it
# under the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import os
import trimesh
import pyrender
import numpy as np
import colorsys
import cv2


class Renderer(object):

    def __init__(self, focal_length=600, center=[256, 256], img_w=512, img_h=512, faces=None,
                 same_mesh_color=False):
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_w,
                                                   viewport_height=img_h,
                                                   point_size=1.0)
        self.camera_center = [center[0], center[1]]
        self.focal_length = focal_length
        self.faces = faces
        self.same_mesh_color = same_mesh_color
        self.use_interaction_color = True
        self.color = [(0.412,0.663,1.0), (1.0,0.749,0.412)]
        self.fix_center_side = [0.1, -0.2, 2.6]
        self.fix_center_top =  [0.0, 0.0, 2.6]
        self.halpe_skeleton_link = [[0,1],[1,3],[0,2],[2,4],[5,18],[6,18],[18,17],[5,7],[7,9],[6,8],[8,10],[5,11],[11,13],[13,15],[6,12],[12,14],[14,16],[11,19],[19,12],[18,19],[15,24],[15,20],[20,22],[16,25],[16,21],[21,23]]
        self.smpl24_skeleton_link = [[0,1],[0,2],[2,5],[5,8],[8,11],[1,4],[4,7],[7,10],[0,3],[3,6],[6,9],[9,12],[9,13],[9,14],[13,16],[16,18],[18,20],[20,22],[14,17],[17,19],[19,21],[21,23],[12,15]]
        self.smpl22_skeleton_link = [[0,1],[0,2],[2,5],[5,8],[8,11],[1,4],[4,7],[7,10],[0,3],[3,6],[6,9],[9,12],[9,13],[9,14],[13,16],[16,18],[18,20],[14,17],[17,19],[19,21],[12,15]]

    def render_front_view(self, verts, bg_img_rgb=None, bg_color=(255, 255, 255, 0)):
        # Create a scene for each image and render all meshes
        scene = pyrender.Scene(bg_color=bg_color, ambient_light=np.ones(3) * 0)
        # Create camera. Camera will always be at [0,0,0]
        camera = pyrender.camera.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                                  cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=np.eye(4))

        # Create light source
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        # for DirectionalLight, only rotation matters
        light_pose = trimesh.transformations.rotation_matrix(np.radians(-45), [1, 0, 0])
        scene.add(light, pose=light_pose)
        light_pose = trimesh.transformations.rotation_matrix(np.radians(45), [0, 1, 0])
        scene.add(light, pose=light_pose)

        # Need to flip x-axis
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        # multiple person
        num_people = len(verts)
        # for every person in the scene
        for n in range(num_people):
            mesh = trimesh.Trimesh(verts[n], self.faces)
            mesh.apply_transform(rot)
            if self.use_interaction_color:
                mesh_color = self.color[n]
            elif self.same_mesh_color:
                mesh_color = colorsys.hsv_to_rgb(0.6, 0.5, 1.0)
            else:
                mesh_color = colorsys.hsv_to_rgb(float(n) / num_people, 0.5, 1.0)
            
                
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.2,
                alphaMode='OPAQUE',
                baseColorFactor=mesh_color)
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material, wireframe=False)
            scene.add(mesh, 'mesh')

        # Alpha channel was not working previously, need to check again
        # Until this is fixed use hack with depth image to get the opacity
        color_rgba, depth_map = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color_rgb = color_rgba[:, :, :3]
        if bg_img_rgb is None:
            return color_rgb
        else:
            mask = depth_map > 0
            bg_img_rgb[mask] = color_rgb[mask]
            return bg_img_rgb

    def render_side_view(self, verts):
        centroid = verts.mean(axis=(0, 1))  # n*6890*3 -> 3
        # print("centroid: ", centroid)
        # make the centroid at the image center (the X and Y coordinates are zeros)
        centroid[:2] = 0
        if self.fix_center_side is None:
            self.fix_center_side = centroid
        aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0][np.newaxis, ...]  # 1*3*3
        pred_vert_arr_side = np.matmul((verts - self.fix_center_side), aroundy) + self.fix_center_side
        side_view = self.render_front_view(pred_vert_arr_side)
        return side_view

    def render_back_view(self, verts):
        centroid = verts.mean(axis=(0, 1))  # n*6890*3 -> 3
        # make the centroid at the image center (the X and Y coordinates are zeros)
        centroid[:2] = 0
        aroundy = cv2.Rodrigues(np.array([np.radians(180.), 0, 0]))[0][np.newaxis, ...]  # 1*3*3
        pred_vert_arr_side = np.matmul((verts - centroid), aroundy) + centroid
        side_view = self.render_front_view(pred_vert_arr_side)
        return side_view

    def render_backside_view(self, verts):
        centroid = verts.mean(axis=(0, 1))  # n*6890*3 -> 3
        # make the centroid at the image center (the X and Y coordinates are zeros)
        centroid[:2] = 0
        aroundy = cv2.Rodrigues(np.array([np.radians(180.), 0, 0]))[0][np.newaxis, ...]  # 1*3*3
        verts = np.matmul((verts - centroid), aroundy) + centroid

        centroid = verts.mean(axis=(0, 1))  # n*6890*3 -> 3
        # make the centroid at the image center (the X and Y coordinates are zeros)
        centroid[:2] = 0
        aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0][np.newaxis, ...]  # 1*3*3
        pred_vert_arr_side = np.matmul((verts - centroid), aroundy) + centroid
        side_view = self.render_front_view(pred_vert_arr_side)
        return side_view
    
    def render_top_view(self, verts):
        centroid = verts.mean(axis=(0, 1))
        centroid[:2] = 0
        if self.fix_center_top is None:
            self.fix_center_top = centroid
        aroundx = cv2.Rodrigues(np.array([np.radians(-90.), 0, 0]))[0][np.newaxis, ...]
        pred_vert_arr_top = np.matmul((verts -  self.fix_center_top), aroundx) +  self.fix_center_top
        top_view = self.render_front_view(pred_vert_arr_top)
        return top_view
    
    def render_joints_front_view(self, joints, bg_img_rgb=None, bg_color=(255, 255, 255, 0)):
        """
        Render 3D joints and connect them with lines according to the skeleton link.
        
        Parameters:
        joints (np.array): Array of 3D joint coordinates.
        bg_img_rgb (np.array): Background image in RGB format.
        bg_color (tuple): Background color in RGBA format.
        
        Returns:
        np.array: Rendered image with joints and skeleton lines.
        """
        scene = pyrender.Scene(bg_color=bg_color, ambient_light=np.ones(3) * 0)
        camera = pyrender.camera.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                                  cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=np.eye(4))

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        light_pose = trimesh.transformations.rotation_matrix(np.radians(-45), [1, 0, 0])
        scene.add(light, pose=light_pose)
        light_pose = trimesh.transformations.rotation_matrix(np.radians(45), [0, 1, 0])
        scene.add(light, pose=light_pose)

        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        joints = np.dot(joints, rot[:3, :3].T)
        joint_num = joints.shape[1]
        is_halpe = joint_num == 26
        is_smpl22 = joint_num == 22
        is_smpl24 = joint_num == 24

        for bs in range(joints.shape[0]):
            for joint in joints[bs]:
                sphere = trimesh.creation.uv_sphere(radius=0.02)
                sphere.apply_translation(joint)
                material = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.2,
                    alphaMode='OPAQUE',
                    baseColorFactor=self.color[bs])
                mesh = pyrender.Mesh.from_trimesh(sphere, material=material)
                scene.add(mesh)
            
            # Draw skeleton lines
            if is_halpe or is_smpl22 or is_smpl24:
                if is_halpe:
                    skeleton_link = self.halpe_skeleton_link
                elif is_smpl22:
                    skeleton_link = self.smpl22_skeleton_link
                elif is_smpl24:
                    skeleton_link = self.smpl24_skeleton_link
                for link in skeleton_link:
                    start_joint = joints[bs][link[0]]
                    end_joint = joints[bs][link[1]]
                    line = trimesh.creation.cylinder(radius=0.005, segment=[start_joint, end_joint])
                    line.apply_translation((start_joint + end_joint) / 2)
                    material = pyrender.MetallicRoughnessMaterial(
                        metallicFactor=0.2,
                        alphaMode='OPAQUE',
                        baseColorFactor=self.color[bs])
                    mesh = pyrender.Mesh.from_trimesh(line, material=material)
                    scene.add(mesh)

        color_rgba, depth_map = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color_rgb = color_rgba[:, :, :3]
        if bg_img_rgb is None:
            return color_rgb
        else:
            mask = depth_map > 0
            bg_img_rgb[mask] = color_rgb[mask]
            return bg_img_rgb
        
    def render_joints_side_view(self, joints):
        centroid = joints.mean(axis=(0, 1))
        centroid[:2] = 0
        if self.fix_center_side is None:
            self.fix_center_side = centroid
        aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0][np.newaxis, ...]
        pred_joint_arr_side = np.matmul((joints - self.fix_center_side), aroundy) + self.fix_center_side
        side_view = self.render_joints_front_view(pred_joint_arr_side)
        return side_view
    
    # def render_joints_back_view(self, joints):
    #     centroid = joints.mean(axis=(0, 1))
    #     centroid[:2] = 0
    #     if self.consistant_center_back is None:
    #         self.consistant_center_back = centroid
    #     aroundy = cv2.Rodrigues(np.array([0, np.radians(180.), 0]))[0][np.newaxis, ...]
    #     pred_joint_arr_back = np.matmul((joints - self.consistant_center_back), aroundy) + self.consistant_center_back
    #     back_view = self.render_joints_front_view(pred_joint_arr_back)
    #     return back_view
    
    def render_joints_top_view(self, joints):
        centroid = joints.mean(axis=(0, 1))
        centroid[:2] = 0
        if self.fix_center_top is None:
            self.fix_center_top = centroid
        aroundx = cv2.Rodrigues(np.array([np.radians(-90.), 0, 0]))[0][np.newaxis, ...]
        pred_joint_arr_top = np.matmul((joints - self.fix_center_top), aroundx) + self.fix_center_top
        top_view = self.render_joints_front_view(pred_joint_arr_top)
        return top_view

    def delete(self):
        """
        Need to delete before creating the renderer next time
        """
        self.renderer.delete()
