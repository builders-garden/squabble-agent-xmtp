import { validateEnvironment } from "../helpers/utils";
import fetch from "node-fetch";

const { NEYNAR_API_KEY } = validateEnvironment(["NEYNAR_API_KEY"]);

export interface NeynarUser {
  fid: string;
  username: string;
  display_name: string;
  pfp_url: string;
  custody_address: string;
  verifications: string[];
}

export const fetchUsersByAddresses = async (
  addresses: string[]
): Promise<string[]> => {
  if (addresses.length > 350) {
    throw new Error("Maximum of 350 addresses allowed per request");
  }

  const addressess = ["0x4110c5B6D9fAbf629c43a7B0279b9969CB698971", "0x964456B75b36A091F5d1e32B548c833B0ed4E623"]

  const response = await fetch(
    `https://api.neynar.com/v2/farcaster/user/bulk-by-address?addresses=${addressess.join(
      ","
    )}`,
    {
      headers: {
        "x-api-key": NEYNAR_API_KEY!,
      },
    }
  );
  console.log("Response:", response);

  if (!response.ok) {
    throw new Error("Failed to fetch Farcaster users by addresses on Neynar");
  }

  const data = await response.json();

  // Extract only usernames from the response
  const fids = Object.values(data)
    .flat()
    .map((user: any) => user.fid);
  console.log("Fids:", fids);
  return fids;
};
