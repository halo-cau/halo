import * as THREE from "three";
import type { Equipment } from "../../data/sceneGraphs";
import { createCableTray } from "./cableTray";
import { createCeilingAc } from "./ceilingAc";
import { createFallback } from "./fallback";
import { createMonitoring } from "./monitoring";
import { createNetworkSwitch } from "./networkSwitch";
import { createPdu } from "./pdu";
import { createServerRack } from "./serverRack";
import { createUps } from "./ups";

export interface EquipmentContext {
  // Room ceiling height — needed by ceiling-mounted equipment
  // (e.g. ceiling AC) so it can anchor flush against the ceiling
  // regardless of the data file's y coordinate.
  roomHeight: number;
}

const DEFAULT_CTX: EquipmentContext = { roomHeight: 3.5 };

export function createEquipmentMesh(
  item: Equipment,
  ctx: EquipmentContext = DEFAULT_CTX,
): THREE.Group {
  let group: THREE.Group;

  switch (item.category) {
    case "server_rack":
      group = createServerRack(item);
      break;
    case "cooling_unit":
    case "ceiling_ac":
      group = createCeilingAc(item, ctx.roomHeight);
      break;
    case "network_switch":
      group = createNetworkSwitch(item);
      break;
    case "ups":
      group = createUps(item);
      break;
    case "pdu":
      group = createPdu(item);
      break;
    case "monitoring":
      group = createMonitoring(item);
      break;
    case "cable_tray":
      group = createCableTray(item);
      break;
    default:
      group = createFallback(item);
  }

  group.traverse((child) => {
    if (child instanceof THREE.Mesh) {
      child.castShadow = true;
      child.receiveShadow = true;
    }
  });

  group.position.set(...item.position);
  group.rotation.set(
    THREE.MathUtils.degToRad(item.rotation[0]),
    THREE.MathUtils.degToRad(item.rotation[1]),
    THREE.MathUtils.degToRad(item.rotation[2]),
  );

  group.userData = {
    id: item.id,
    label: item.label,
    category: item.category,
    heatOutput: item.heatOutput,
  };

  return group;
}
